"""Execute notebook controls and BYOD paths without model downloads.

Model doubles prove orchestration, not real-model or hosted-runtime readiness.
"""
import ast
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
from packaging.version import InvalidVersion, Version
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb'
LABELS = ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific report',
          'scientific publication', 'specification', 'file folder', 'news article', 'budget',
          'invoice', 'presentation', 'questionnaire', 'resume', 'memo']


def cell(key):
    """Source of a cell by index or by its stable cell id."""
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    if isinstance(key, str):
        return ''.join(next(c for c in cells if c['id'] == key)['source'])
    return ''.join(cells[key]['source'])


def helpers(index, names, namespace):
    tree = ast.parse(cell(index))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, 'notebook-helper', 'exec'), namespace)


def namespace(tmp_path, monkeypatch, vector=None):
    monkeypatch.chdir(tmp_path)
    calls = []
    if vector is None:
        vector = np.full(16, 0.2 / 15)
        vector[0] = 0.8

    def predict(records, batch_size):
        calls.append(len(records))
        scores = np.repeat(np.asarray(vector, dtype=float)[None, :], len(records), axis=0)
        return np.log(scores), scores, [0.0], {}

    shown = []
    ns = dict(Path=Path, Image=Image, ImageDraw=ImageDraw, math=math, np=np, csv=csv, json=json,
              hashlib=hashlib, display=shown.append, shown=shown,
              LABELS=LABELS, LABEL_TO_ID={v: i for i, v in enumerate(LABELS)},
              MIN_IMAGE_SIDE=32, MAX_IMAGE_SIDE=4096, MAX_PIXELS=32_000_000,
              BATCH_SIZE=16, USE_BYOD=False, RUN_ROBUSTNESS=True, BYOD_LABELS_PATH='',
              OUTPUT_DIR=str(tmp_path/'outputs'), MODEL_ID='test-double',
              MODEL_REVISION='fixed-test-revision', converted_sha256='test-conversion',
              RUNTIME={'model_double': True}, batched_predict=predict)
    helpers(9, {'sha256_file'}, ns)
    helpers('18aafd5c', {'make_gallery'}, ns)
    helpers('b485b2eb', {'rank_classes'}, ns)
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


# --- Review fixes (DIT-M01..M03, DIT-m01..m04). Model doubles only: not real-model evidence.

def tied_vector(first=2, second=3):
    vector = (np.arange(16, dtype=np.float32) + 1) / 1000
    vector[[first, second]] = 1.0
    return vector / vector.sum()


def notebook_code():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return [(c['id'], ''.join(c['source'])) for c in notebook['cells'] if c['cell_type'] == 'code']


@pytest.mark.parametrize(('row', 'expected'), [
    ([0.1, 0.7, 0.2], [1, 2, 0]),
    ([0.1, 0.4, 0.1, 0.4], [1, 3, 0, 2]),
    ([0.25] * 4, [0, 1, 2, 3]),
    ([0.2, 0.3, 0.05, 0.25, 0.2], [1, 3, 0, 4, 2]),
])
def test_notebook_rank_classes_tie_rule(row, expected):
    ns = dict(np=np)
    helpers('b485b2eb', {'rank_classes'}, ns)
    order = ns['rank_classes'](np.asarray(row, dtype=np.float32))[0]
    assert order.tolist() == expected and int(order[0]) == int(np.argmax(row))


def test_ranking_is_shared_by_every_code_cell():
    for cid, source in notebook_code():
        assert 'argsort(' not in source and '.argmax(' not in source, cid


@pytest.mark.parametrize('gold', ['email', 'handwritten'])
def test_byod_tie_exports_agree_with_accuracy(tmp_path, monkeypatch, gold):
    ns, calls = namespace(tmp_path, monkeypatch, vector=tied_vector(2, 3))
    image = tmp_path/'page.png'
    image.write_bytes(image_bytes())
    labels = tmp_path/'labels.csv'
    labels.write_text(f'filename,label\npage.png,{gold}\n')
    ns.update(USE_BYOD=True, BYOD_PATH=str(image), BYOD_LABELS_PATH=str(labels))
    exec(cell('907b911c'), ns)
    result = json.loads((tmp_path/'outputs/byod_results.json').read_text())
    with (tmp_path/'outputs/byod_predictions.csv').open() as handle:
        (row,) = list(csv.DictReader(handle))
    with (tmp_path/'outputs/byod_class_scores.csv').open() as handle:
        ranks = {r['class_label']: int(r['rank']) for r in csv.DictReader(handle)}
    assert row['predicted_label'] == 'email' and ranks['email'] == 1 and ranks['handwritten'] == 2
    assert result['accuracy'] == (1.0 if gold == 'email' else 0.0)
    assert row['correct'] == str(gold == 'email')
    assert sorted(ranks.values()) == list(range(1, 17))


def tiff_bytes(frames):
    stream = io.BytesIO()
    pages = [Image.new('RGB', (64, 96), color) for color in ('white', 'black', 'red')[:frames]]
    pages[0].save(stream, format='TIFF', save_all=True, append_images=pages[1:])
    return stream.getvalue()


def test_multi_page_tiff_rejected_before_inference(tmp_path, monkeypatch):
    ns, calls = namespace(tmp_path, monkeypatch)
    path = tmp_path/'scan.tiff'
    path.write_bytes(tiff_bytes(2))
    ns.update(USE_BYOD=True, BYOD_PATH=str(path))
    with pytest.raises(ValueError, match='scan.tiff: contains 2 pages'):
        exec(cell('907b911c'), ns)
    assert calls == [] and not (tmp_path/'outputs').exists()


def test_single_page_tiff_accepted(tmp_path, monkeypatch):
    ns, calls = namespace(tmp_path, monkeypatch)
    path = tmp_path/'scan.tif'
    path.write_bytes(tiff_bytes(1))
    ns.update(USE_BYOD=True, BYOD_PATH=str(path))
    exec(cell('907b911c'), ns)
    assert calls == [1] and ns['byod_result']['images'] == 1


def test_phone_mpo_jpeg_is_one_page(tmp_path, monkeypatch):
    # Phone cameras write MPO JPEGs whose second frame is a preview, not another page.
    ns, calls = namespace(tmp_path, monkeypatch)
    path = tmp_path/'photo.jpg'
    Image.new('RGB', (64, 96), 'white').save(
        path, format='MPO', save_all=True, append_images=[Image.new('RGB', (64, 96), 'black')])
    with Image.open(path) as check:
        assert check.format == 'MPO' and check.n_frames == 2
    ns.update(USE_BYOD=True, BYOD_PATH=str(path))
    exec(cell('907b911c'), ns)
    assert calls == [1] and ns['byod_records'][0]['image'].getpixel((0, 0)) == (255, 255, 255)


def test_oversized_page_rejected_from_header(tmp_path, monkeypatch):
    ns, calls = namespace(tmp_path, monkeypatch)
    path = tmp_path/'wide.png'
    Image.new('L', (5000, 40), 255).save(path)
    ns.update(USE_BYOD=True, BYOD_PATH=str(path))
    with pytest.raises(ValueError, match='side outside'):
        exec(cell('907b911c'), ns)
    assert calls == []


@pytest.mark.parametrize(('text', 'message'), [
    ('filename,label,label\npage.png,letter,email\n', 'duplicate column names'),
    ('filename,label\npage.png,letter,extra\n', 'exactly 2 fields'),
    ('filename,label\npage.png\n', 'exactly 2 fields'),
])
def test_malformed_label_csv_schema_refused(tmp_path, monkeypatch, text, message):
    ns, calls = namespace(tmp_path, monkeypatch)
    image = tmp_path/'page.png'
    image.write_bytes(image_bytes())
    labels = tmp_path/'labels.csv'
    labels.write_text(text)
    ns.update(USE_BYOD=True, BYOD_PATH=str(image), BYOD_LABELS_PATH=str(labels))
    with pytest.raises(ValueError, match=message):
        exec(cell('907b911c'), ns)
    assert calls == [] and not (tmp_path/'outputs').exists()


@pytest.mark.parametrize('labelled', [False, True])
def test_byod_prints_predictions_and_class_support(tmp_path, monkeypatch, capsys, labelled):
    ns, _calls = namespace(tmp_path, monkeypatch)
    image = tmp_path/'my-page.png'
    image.write_bytes(image_bytes())
    ns.update(USE_BYOD=True, BYOD_PATH=str(image))
    if labelled:
        labels = tmp_path/'labels.csv'
        labels.write_text('filename,label\nmy-page.png,letter\n')
        ns['BYOD_LABELS_PATH'] = str(labels)
    exec(cell('907b911c'), ns)
    out = capsys.readouterr().out
    line = next(t for t in out.splitlines() if t.startswith('my-page.png'))
    assert 'letter' in line and '0.800' in line
    assert 'byod_class_scores.csv' in out
    result = json.loads((tmp_path/'outputs/byod_results.json').read_text())
    if labelled:
        assert result['accuracy'] == 1.0 and result['macro_f1'] == pytest.approx(0.0625)
        assert result['classes_with_support'] == ['letter'] and len(result['per_class']) == 16
        assert 'averages all 16 fixed classes; 1 of 16' in out
    else:
        assert 'per_class' not in result and 'Macro F1' not in out


def robust_namespace(tmp_path, monkeypatch, changed_variant=None):
    ns, calls = namespace(tmp_path, monkeypatch)

    def predict(records, batch_size):
        calls.append(len(records))
        scores = np.full((len(records), 16), 0.2 / 15)
        for i, record in enumerate(records):
            scores[i, 1 if record['variant'] == changed_variant else 0] = 0.8
        return np.log(scores), scores, [0.0], {}

    base = [{'image_id': f'page-{i}', 'image': Image.new('RGB', (40, 48)), 'label': 'letter', 'label_id': 0}
            for i in range(2)]
    ns.update(batched_predict=predict, robustness_base=base)
    return ns, calls


@pytest.mark.parametrize('changed_variant', [None, 'rotate90', 'center_crop'])
def test_robustness_shows_changed_pages_and_one_pair(tmp_path, monkeypatch, capsys, changed_variant):
    ns, calls = robust_namespace(tmp_path, monkeypatch, changed_variant)
    exec(cell('3891d68d'), ns)
    out = capsys.readouterr().out
    assert calls == [10] and len(ns['shown']) == 1
    if changed_variant is None:
        assert ns['changed_rows'] == [] and 'none: every page kept' in out
        assert 'page-0 (gold: letter), rotate90' in out
    else:
        assert [r['image_id'] for r in ns['changed_rows']] == ['page-0', 'page-1']
        assert f'page-0                   letter                 {changed_variant}' in out
        assert f'page-0 (gold: letter), {changed_variant}' in out


def test_robustness_disabled_displays_nothing(tmp_path, monkeypatch):
    ns, calls = robust_namespace(tmp_path, monkeypatch)
    ns['RUN_ROBUSTNESS'] = False
    exec(cell('3891d68d'), ns)
    assert calls == [] and ns['shown'] == [] and ns['changed_rows'] == []


def test_galleries_displayed_and_stale_error_gallery_removed(tmp_path, monkeypatch):
    ns, _calls = namespace(tmp_path, monkeypatch)
    records = [{'image_id': f'p{i}', 'image': Image.new('RGB', (40, 48), 'white')} for i in range(16)]
    rows = [{'image_id': f'p{i}', 'true_label': LABELS[i], 'predicted_label': LABELS[i],
             'top1_score': 0.9} for i in range(16)]
    wrong = dict(rows[3], predicted_label='scientific publication', top1_score=0.7)
    ns.update(sample_records=records, prediction_rows=rows, robustness_base=records,
              high_conf_wrong=[wrong])
    exec(cell('100a15ce'), ns)
    stale = tmp_path/'outputs/galleries/high_confidence_errors.png'
    assert stale.is_file() and len(ns['shown']) == 2
    ns['shown'].clear()
    ns['high_conf_wrong'] = []
    exec(cell('100a15ce'), ns)
    assert not stale.exists() and ns['error_gallery_path'] is None and len(ns['shown']) == 1
    assert (tmp_path/'outputs/galleries/class_gallery.png').is_file()


def test_sample_preview_displayed_before_classification(tmp_path, monkeypatch):
    ns, _calls = namespace(tmp_path, monkeypatch)
    dataset = [{'image': Image.new('RGB', (40, 48), (i % 256, i // 256, 7)), 'label': i % 16}
               for i in range(16 * 21)]
    ns.update(dataset_test=dataset, SAMPLE_SEED=42, SAMPLE_PER_CLASS=20)
    exec(cell('18aafd5c'), ns)
    assert len(ns['sample_records']) == 320 and len(ns['shown']) == 1
    assert [r['label_id'] for r in ns['preview_records']] == list(range(16))


def test_provenance_and_conclusion_wording_match_evidence_boundary():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    spec = notebook['metadata']['dimer']['notebook_spec']
    export = cell('8be87936')
    assert f'"notebook_spec": "{spec}"' in export and '"evaluation_limitations"' in export
    assert 'out_dir.iterdir()' not in export
    assert 'held-out classification result' not in cell('guided-03')
    dataset = cell('ca003003')
    revision = notebook['metadata']['dimer']['dataset']['revision']
    assert len(revision) == 40 and f'DATASET_REVISION = "{revision}"' in dataset
    assert 'repo_info' not in dataset
