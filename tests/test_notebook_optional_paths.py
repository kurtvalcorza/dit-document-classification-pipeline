"""Execute notebook controls and BYOD paths without model downloads.

Model doubles prove orchestration, not real-model or hosted-runtime readiness.
"""
import ast
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from packaging.version import InvalidVersion, Version
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb'
LABELS = ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific report',
          'scientific publication', 'specification', 'file folder', 'news article', 'budget',
          'invoice', 'presentation', 'questionnaire', 'resume', 'memo']


def cell(index):
    return ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][index]['source'])


def helpers(index, names, namespace):
    tree = ast.parse(cell(index))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, 'notebook-helper', 'exec'), namespace)


def namespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def predict(records, batch_size):
        calls.append(len(records))
        scores = np.zeros((len(records), 16))
        scores[:, 0] = 0.8
        scores[:, 1:] = 0.2 / 15
        return np.log(scores), scores, [0.0], {}

    ns = dict(Path=Path, Image=Image, np=np, csv=csv, json=json, hashlib=hashlib,
              LABELS=LABELS, LABEL_TO_ID={v: i for i, v in enumerate(LABELS)},
              MIN_IMAGE_SIDE=32, MAX_IMAGE_SIDE=4096, MAX_PIXELS=32_000_000,
              BATCH_SIZE=16, USE_BYOD=False, RUN_ROBUSTNESS=True, BYOD_LABELS_PATH='',
              OUTPUT_DIR=str(tmp_path/'outputs'), MODEL_ID='test-double',
              MODEL_REVISION='fixed-test-revision', converted_sha256='test-conversion',
              RUNTIME={'model_double': True}, batched_predict=predict)
    helpers(9, {'sha256_file'}, ns)
    helpers(23, {'confusion_matrix_np', 'per_class_metrics_from_cm'}, ns)
    helpers(37, {'write_csv'}, ns)
    exec(cell(39), ns)
    return ns, calls


def image_bytes(color='white'):
    stream = io.BytesIO()
    Image.new('RGB', (40, 48), color).save(stream, format='PNG')
    return stream.getvalue()


def archive(path, members):
    with zipfile.ZipFile(path, 'w') as handle:
        for name, data in members:
            handle.writestr(name, data)
    return path


@pytest.mark.parametrize(('observed', 'expected'), [
    ('2.14.0+cu128', True), ('2.14.0', True), ('2.14.0rc1', False),
    ('2.14.0.post1', False), ('2.14.1', False), (None, False), ('invalid', False),
])
def test_runtime_public_version(observed, expected):
    ns = dict(Version=Version, InvalidVersion=InvalidVersion)
    helpers(7, {'matches_public_version'}, ns)
    assert ns['matches_public_version'](observed, '2.14.0') is expected


def test_robustness_gate_and_header_only_export(tmp_path, monkeypatch):
    ns, calls = namespace(tmp_path, monkeypatch)
    ns.update(RUN_ROBUSTNESS=False, robustness_base=object())
    exec(cell(31), ns)
    assert calls == [] and ns['robustness_rows'] == [] and ns['robustness_summary'] == {}
    out = tmp_path/'robustness.csv'
    ns['write_csv'](out, ns['robustness_rows'], ['image_id', 'variant'])
    assert out.read_text().strip() == 'image_id,variant'
    ns.update(RUN_ROBUSTNESS=True, robustness_base=[
        {'image_id': 'page.png', 'image': Image.new('RGB', (40, 48)), 'label': 'letter', 'label_id': 0}])
    exec(cell(31), ns)
    assert calls == [5] and len(ns['robustness_rows']) == 5
    assert ns['robustness_summary']['original']['accuracy'] == 1.0


def test_zip_duplicate_refused_and_attempts_do_not_mix(tmp_path, monkeypatch):
    ns, calls = namespace(tmp_path, monkeypatch)
    members = [('a/page.png', image_bytes()), ('b/page.png', image_bytes('red'))]
    path = archive(tmp_path/'duplicate.zip', members)
    with pytest.raises(ValueError, match='Duplicate image basename'):
        ns['safe_extract_images'](path, tmp_path/'staging')
    extract, staging = ns['safe_extract_images'], tmp_path/'staging'
    first = extract(archive(tmp_path/'one.zip', [('first.png', image_bytes())]), staging)
    second = extract(archive(tmp_path/'two.zip', [('second.png', image_bytes())]), staging)
    assert first != second
    assert [p.name for p in first.iterdir()] == ['first.png']
    assert [p.name for p in second.iterdir()] == ['second.png']
    assert calls == []


@pytest.mark.parametrize('labelled', [False, True])
def test_byod_full_optional_flow_exports_unique_ids(tmp_path, monkeypatch, labelled):
    ns, calls = namespace(tmp_path, monkeypatch)
    path = archive(tmp_path/'pages.zip', [('page.png', image_bytes()), ('page.jpg', image_bytes('red'))])
    ns.update(USE_BYOD=True, BYOD_PATH=str(path))
    if labelled:
        labels = tmp_path/'labels.csv'
        labels.write_text('filename,label\npage.png,letter\npage.jpg,form\n')
        ns['BYOD_LABELS_PATH'] = str(labels)
    exec(cell(39), ns)
    result = json.loads((tmp_path/'outputs/byod_results.json').read_text())
    assert result['evaluation_verdict'] == ('measured' if labelled else 'not-measurable')
    assert set(result['provenance']['image_sha256']) == {'page.png', 'page.jpg'}
    assert result['provenance']['source_zip_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with (tmp_path/'outputs/byod_predictions.csv').open() as handle:
        assert {r['image_id'] for r in csv.DictReader(handle)} == {'page.png', 'page.jpg'}
    assert calls == [2]
    if labelled:
        assert result['accuracy'] == 0.5


@pytest.mark.parametrize(('rows', 'message'), [
    ('page.png,letter\npage.png,form\n', 'Duplicate'),
    ('other.png,letter\n', 'missing='),
    ('page.png,letter\nextra.png,form\n', 'extra='),
    ('page.png,unknown\n', 'Unknown RVL-CDIP'),
])
def test_invalid_labels_fail_before_inference(tmp_path, monkeypatch, rows, message):
    ns, calls = namespace(tmp_path, monkeypatch)
    image = tmp_path/'page.png'
    image.write_bytes(image_bytes())
    labels = tmp_path/'labels.csv'
    labels.write_text('filename,label\n'+rows)
    ns.update(USE_BYOD=True, BYOD_PATH=str(image), BYOD_LABELS_PATH=str(labels))
    with pytest.raises(ValueError, match=message):
        exec(cell(39), ns)
    assert calls == [] and not (tmp_path/'outputs').exists()
