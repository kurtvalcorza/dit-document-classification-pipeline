"""Regression tests for review round 2026-10-12 (DIT2-M1, DIT2-m1..m6).

Static source checks plus notebook cells executed with model doubles.
Not real-model or hosted-runtime evidence.
"""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from test_notebook_optional_paths import (
    LABELS,
    NOTEBOOK,
    archive,
    assert_refused,
    cell,
    image_bytes,
    namespace,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE_FILE = ROOT / 'tools' / 'document_classification_workshop.py'


def markdown():
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    return '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'markdown')


# ---- DIT2-M1: label noise, counts out of 20, error-gallery activity --------------------------------------

# Wording checks: these establish that the guidance is present, not that its interpretation is valid.

def test_wording_confusion_guidance_treats_label_noise_as_background():
    text = cell('601e66d3')
    assert 'out of 20' in text and '1 or 2 pages is an anecdote' in text
    assert '8.1%' in text and 'arXiv:2306.12550' in text and 'Larson, Lim and Leach (2023)' in text
    assert '1,000 pages per class (16,000) sampled from the original RVL-CDIP test split' in text
    assert ('This estimate cannot be used to correct our accuracy or determine which disagreements are label'
            ' errors.') in text
    assert 'agreement with the supplied labels' in text and 'cannot decide' in text
    assert 'within that label-error rate' not in text and 'near or below that figure' not in text
    assert 'out of 20 pages' in cell('52854116')


def test_wording_label_review_activity_is_tentative_and_has_uncertainty_options():
    text = cell('1d4d0e0d')
    assert 'tentative visual judgement, not label adjudication' in text
    assert '**another class may fit**' in text and '**cannot determine**' in text
    assert 'cannot estimate the overall label-noise rate' in text
    assert 'pages/sample/<image_id>.png' in text and 'Short class descriptions' in text
    assert 'close to what this benchmark can measure' not in markdown()
    conclusion = cell('guided-03')
    assert 'count out of 20 pages' in conclusion and 'as a tentative judgement' in conclusion
    assert 'cannot determine' in conclusion


def test_confusion_pairs_print_their_denominator(capsys):
    cm = np.eye(16, dtype=int) * 20
    cm[3, 3], cm[3, 13] = 18, 2
    off = [(2, 3, 13)]
    exec(cell('474bc879'), {'cm': cm, 'LABELS': LABELS, 'off_diagonal': off})
    out = capsys.readouterr().out
    assert 'handwritten -> questionnaire: 2 of 20 pages' in out
    assert 'the gold label itself may be wrong' in out


# ---- DIT2-m1: derivation notebook, resize, test/train overlap ---------------------------------------------

def test_evaluation_source_cites_derivation_notebook_and_overlap():
    text = cell('bd966b5d')
    assert 'colab.research.google.com/drive/1CuER4BrT8uQj0nlPX0Fcek_zYqyaOvGA' in text
    assert 'was not\ninspected' not in text and 'not inspected for this tutorial' in text
    assert 'does not exceed 1,000 pixels' in text
    assert 'de-duplicated against the RVL-CDIP training split' in text
    assert 'drawn directly from the official test split without changing' not in text
    export = cell('8be87936')
    assert 'RVL-CDIP test and train splits overlap' in export
    assert 'derivation notebook was not inspected' in export


# ---- DIT2-m2: robustness counts, baseline gap, example correct at baseline --------------------------------

def _robustness(tmp_path, monkeypatch, wrong_at_original):
    ns, calls = namespace(tmp_path, monkeypatch)

    def predict(records, batch_size):
        calls.append(len(records))
        scores = np.full((len(records), 16), 0.2 / 15)
        for i, record in enumerate(records):
            if record['variant'] == 'rotate90':
                predicted = 2  # every page changes class under rotation
            elif record['image_id'] in wrong_at_original and record['variant'] != 'center_crop':
                predicted = 1
            else:
                predicted = 0
            scores[i, predicted] = 0.8
        return np.log(scores), scores, [0.0], {}

    base = [{'image_id': f'page-{i}', 'image': Image.new('RGB', (40, 48)), 'label': 'letter', 'label_id': 0}
            for i in range(4)]
    ns.update(batched_predict=predict, robustness_base=base)
    return ns


def test_robustness_prints_counts_and_pages_wrong_at_baseline(tmp_path, monkeypatch, capsys):
    ns = _robustness(tmp_path, monkeypatch, {'page-0'})
    exec(cell('3891d68d'), ns)
    out = capsys.readouterr().out
    assert 'one page changes accuracy or stability by 0.2500' in out
    assert re.search(r'original\s+3/4\s+0\.750\s+4/4', out)
    assert re.search(r'rotate90\s+0/4\s+0\.000\s+0/4', out)
    assert 'already misclassified with no transformation: 1 of 4' in out and 'page-0: gold=letter' in out
    # The summary values themselves are unchanged by the display.
    assert ns['robustness_summary']['original']['accuracy'] == 0.75


def test_before_after_example_prefers_a_page_correct_at_baseline(tmp_path, monkeypatch, capsys):
    ns = _robustness(tmp_path, monkeypatch, {'page-0'})
    exec(cell('3891d68d'), ns)
    out = capsys.readouterr().out
    # page-0 changes first under rotate90 but was already wrong; page-1 was correct and broke.
    assert 'Before/after example: page-1 (gold: letter), rotate90; original prediction correct' in out


def test_before_after_example_falls_back_when_every_changed_page_was_wrong(tmp_path, monkeypatch, capsys):
    ns = _robustness(tmp_path, monkeypatch, {'page-0', 'page-1', 'page-2', 'page-3'})
    exec(cell('3891d68d'), ns)
    out = capsys.readouterr().out
    assert 'Before/after example: page-0 (gold: letter), rotate90; original prediction already wrong' in out


def test_granularity_and_baseline_gap_are_explained():
    text = cell('3f050f2d')
    assert 'one page changes accuracy by 0.0625' in text and 'not the 320-page accuracy' in text
    assert 'whose original prediction was correct' in text
    assert '__/16' in cell('guided-01')
    # CX5 (wording): no change in binary correctness, not "shows nothing"; the example is outcome-selected.
    assert 'shows nothing about that transformation' not in text
    assert 'no change in binary correctness' in text and 'selected by its outcome' in text


def test_paired_outcomes_probe_kinds_and_outcome_selected_example(tmp_path, monkeypatch, capsys):
    # CX4/CX5: paired counts from existing predictions; crop labelled as a partial-page probe (display only).
    ns = _robustness(tmp_path, monkeypatch, {'page-0'})
    exec(cell('3891d68d'), ns)
    out = capsys.readouterr().out
    assert re.search(r'rotate90\s+3\s+0\s+1\s+0\n', out)       # correct->wrong, wrong->correct, other, same
    assert re.search(r'center_crop\s+0\s+1\s+0\s+3\n', out)
    assert re.search(r'rotate180\s+0\s+0\s+0\s+4\n', out)
    assert re.search(r'center_crop .*partial page', out) and re.search(r'rotate90 .*orientation', out)
    assert 'about 64% of the page area' in out
    assert 'selected by its outcome' in out
    assert ns['robustness_summary']['center_crop']['accuracy'] == 1.0


def test_wording_crop_and_selective_classification():
    text = cell('c0ca8456')
    assert '64% of the page area' in text and 'label may no longer be observable' in text
    assert 'Each probe keeps the same ground-truth class' not in text
    selective = cell('ae1644b2')
    assert 'exploratory selective-classification analysis' in selective
    assert 'calibration behavior' not in selective and 'independent' in selective


def test_reference_values_for_each_metric(capsys):
    # CX6: random top-3 is 3/16 and the always-majority macro F1 is 1/136; the existing values do not change.
    true_ids = np.repeat(np.arange(16), 20)
    from collections import Counter
    ns = {'np': np, 'LABELS': LABELS, 'counts': Counter(true_ids.tolist()),
          'sample_records': [{'label_id': int(i)} for i in true_ids]}
    exec(cell('bbc5b89f'), ns)
    assert ns['random_accuracy_expectation'] == 1 / 16 and ns['majority_accuracy'] == 1 / 16
    assert ns['random_top3_expectation'] == pytest.approx(3 / 16)
    assert ns['majority_macro_f1'] == pytest.approx(1 / 136)
    assert "'majority_macro_f1'" in capsys.readouterr().out
    assert '"reference_values"' in cell('8be87936') and 'random top-3' in cell('d60ee8bb')


# ---- DIT2-m3: one refusal line, no traceback, stale exports cleared, genuine failures still raise ---------

def test_refusal_after_an_accepted_attempt_leaves_no_byod_exports(tmp_path, monkeypatch, capsys):
    ns, calls = namespace(tmp_path, monkeypatch)
    good = tmp_path/'good.png'
    good.write_bytes(image_bytes())
    ns.update(USE_BYOD=True, BYOD_PATH=str(good))
    exec(cell('907b911c'), ns)
    assert len(list(tmp_path.glob('outputs/byod_*'))) == 3 and calls == [1]
    bad = tmp_path/'scan.tiff'
    pages = [Image.new('RGB', (64, 96), c) for c in ('white', 'black')]
    pages[0].save(bad, format='TIFF', save_all=True, append_images=pages[1:])
    ns['BYOD_PATH'] = str(bad)
    calls.clear()
    capsys.readouterr()
    exec(cell('907b911c'), ns)
    assert_refused(ns, calls, capsys, 'scan.tiff: contains 2 pages', tmp_path)


@pytest.mark.parametrize(('make', 'message'), [
    (lambda d: '', 'BYOD_PATH is empty'),
    (lambda d: str(d/'missing.png'), 'is not a page image'),
    (lambda d: (d/'broken.png').write_bytes(b'not an image') and str(d/'broken.png'), 'not a readable image'),
    (lambda d: (d/'cut.png').write_bytes(image_bytes()[:60]) and str(d/'cut.png'), 'broken'),
    (lambda d: (d/'pages.zip').write_bytes(b'PK no zip') and str(d/'pages.zip'), 'not a readable ZIP'),
])
def test_bad_inputs_are_refused_with_one_line(tmp_path, monkeypatch, capsys, make, message):
    ns, calls = namespace(tmp_path, monkeypatch)
    ns.update(USE_BYOD=True, BYOD_PATH=make(tmp_path))
    capsys.readouterr()
    exec(cell('907b911c'), ns)
    out = capsys.readouterr().out
    refusals = [line for line in out.splitlines() if line.startswith('Refused:')]
    assert len(refusals) == 1 and 'Traceback' not in out and calls == []
    if message != 'broken':  # a truncated PNG is refused when its header or its data cannot be decoded
        assert message in refusals[0], out


@pytest.mark.parametrize(('content', 'message'), [
    (None, 'is not a file'),
    (b'filename,label\n\xff\xfe page.png,letter\n', 'cannot be read as a UTF-8 CSV'),
])
def test_unreadable_labels_file_is_refused(tmp_path, monkeypatch, capsys, content, message):
    ns, calls = namespace(tmp_path, monkeypatch)
    image = tmp_path/'page.png'
    image.write_bytes(image_bytes())
    labels = tmp_path/'labels.csv'
    if content is not None:
        labels.write_bytes(content)
    ns.update(USE_BYOD=True, BYOD_PATH=str(image), BYOD_LABELS_PATH=str(labels))
    capsys.readouterr()
    exec(cell('907b911c'), ns)
    assert_refused(ns, calls, capsys, message, tmp_path)


def test_zip_member_rules_are_refusals(tmp_path, monkeypatch, capsys):
    ns, calls = namespace(tmp_path, monkeypatch)
    path = archive(tmp_path/'dup.zip', [('a/page.png', image_bytes()), ('b/page.png', image_bytes('red'))])
    ns.update(USE_BYOD=True, BYOD_PATH=str(path))
    capsys.readouterr()
    exec(cell('907b911c'), ns)
    assert_refused(ns, calls, capsys, 'Duplicate image basename in ZIP: page.png', tmp_path)
    assert issubclass(ns['BYODInputError'], ValueError)


def test_a_genuine_failure_still_raises(tmp_path, monkeypatch):
    ns, _calls = namespace(tmp_path, monkeypatch)
    image = tmp_path/'page.png'
    image.write_bytes(image_bytes())

    def crash(records, batch_size):
        raise RuntimeError('predict failed with exit 1')

    ns.update(USE_BYOD=True, BYOD_PATH=str(image), batched_predict=crash)
    with pytest.raises(RuntimeError, match='predict failed'):
        exec(cell('907b911c'), ns)
    assert ns['byod_refusal'] is None


def test_terminal_cell_and_run_stage_follow_the_refusal_contract():
    terminal = cell('d60ee8bb')
    assert 'if USE_BYOD and byod_result is not None:' in terminal
    assert 'BYOD: refused in Section 20' in terminal
    runtime = cell('442373b0')
    assert 'print(log.read_text' not in runtime and 'the full log is {log}' in runtime
    assert 'Refused:' in cell('guided-04') and 'Refused:' in cell('525cb60f')


# ---- DIT2-m4: stage lines are not repeated by the kernel; library messages explained ----------------------

def test_stage_lines_are_labelled_and_messages_explained():
    stage = STAGE_FILE.read_text(encoding='utf-8')
    assert 'print({"sample_images"' not in stage and 'print({"pages"' not in stage
    assert 'sample stage: ' in stage and 'predict stage: ' in stage
    text = cell('6fdf13b3')
    assert 'Using a slow image processor' in text and "'reduce_labels'" in text and 'expected and' in text


def test_stage_sample_and_predict_differ_only_in_their_final_print():
    functions = {n.name: n for n in ast.parse(STAGE_FILE.read_text(encoding='utf-8')).body
                 if isinstance(n, ast.FunctionDef)}
    for name in ('stage_sample', 'stage_predict'):
        last = functions[name].body[-1]
        assert isinstance(last, ast.Expr) and last.value.func.id == 'print'
        assert not any(isinstance(n, ast.Dict) for n in ast.walk(last))


# ---- DIT2-m5: guided layer specific to this notebook; observable objectives --------------------------------

def test_guided_layer_has_no_template_phrases_and_states_objectives():
    text = markdown()
    for phrase in ('model(s)', 'multi-model', 'and/or', 'model/system', 'cost/complexity'):
        assert phrase not in text, phrase
    opening = cell('11007f70')
    assert 'By the end you will be able to:' in opening
    for verb in ('**explain**', '**verify**', '**compare**', '**interpret**', '**identify**'):
        assert verb in opening, verb
    assert 'The notebook covers:' not in opening


# ---- DIT2-m6: kind, use notice, stale hosted-execution note; spec version kept ----------------------------

def test_kind_notice_and_revision_log():
    meta = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['metadata']['dimer']
    keys = list(meta)
    assert meta['notebook_kind'] == 'WORKSHOP'
    assert keys.index('notebook_kind') == keys.index('notebook_mode') + 1
    assert meta['notebook_spec'] == '2.2'
    changes = [r['change'] for r in meta['review_revisions']]
    assert 'review round 2026-10-12 (DIT2-M1, DIT2-m1..m6)' in changes
    assert changes[-1] == 'peer-review follow-up (CX1-CX10)'
    assert 'pending' not in meta['review_revisions'][0]['hosted_execution']
    opening = cell('11007f70')
    assert '**Notebook kind:** `WORKSHOP`' in opening
    notice = opening.index('> **⚠️ Notice.** For research, training and evaluation only.')
    assert notice < opening.index('By the end you will be able to:')
    assert 'Data you bring stays in your runtime' in opening and 'accept no liability' in opening


# ---- Peer-review follow-up: training exposure (CX1), provenance fields (CX7), label mapping (CX8) ----------

def test_wording_training_exposure_is_split_by_stage():
    text = cell('bd966b5d')
    assert 'Overlap with pre-training is therefore certain' not in text
    assert '*Parent corpus.*' in text and '**possible but not verified**' in text and '**unknown**' in text
    assert 'not shown to be held out from either training stage' in text
    export = cell('8be87936')
    assert 'and so overlaps DiT' not in export and 'possible but unverified' in export
    assert '"unsupervised_pretraining": "possible, unverified"' in export
    assert 'not shown to be held out from either training stage' in cell('guided-03')
    assert "were part of DiT's IIT-CDIP pre-training corpus" not in cell('8cf21741')


def test_manifest_keeps_source_bytes_key_and_adds_rgb_png_hash():
    export = cell('8be87936')
    assert '"source_bytes_sha256":r["source_bytes_sha256"]' in export
    assert '"rgb_png_sha256":sha256_file(r["png"])' in export
    assert '"sample_manifest_fields"' in export and 're-encoded in RGB' in export


class FakeFeature:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class FakeDataset(list):
    def __init__(self, labels, names):
        super().__init__({'label': label} for label in labels)
        self.features = {'label': FakeFeature(names=names), 'image': FakeFeature(decode=True)}

    def __getitem__(self, key):
        if key == 'label':
            return [row['label'] for row in self]
        return list.__getitem__(self, key)


def _dataset_stage(tmp_path, monkeypatch, names):
    import document_classification_workshop as stages
    data = FakeDataset([i % 16 for i in range(992)], names)
    monkeypatch.setattr(stages, 'CONFIG', {'dataset_id': 'fake', 'dataset_revision': '0' * 40})
    monkeypatch.setattr(stages, 'WORK', tmp_path)
    monkeypatch.setattr(stages, 'load_pinned_test_split', lambda: data)
    stages.stage_dataset()
    return json.loads((tmp_path / 'state' / 'dataset.json').read_text(encoding='utf-8'))


def test_dataset_stage_exports_label_names_and_full_split_counts(tmp_path, monkeypatch, capsys):
    record = _dataset_stage(tmp_path, monkeypatch, list(LABELS))
    assert record['label_names'] == LABELS and record['label_names_match_class_order'] is True
    assert record['test_split_class_counts'] == {name: 62 for name in LABELS}
    assert record['image_feature_decodes'] is True and record['test_rows'] == 992
    assert 'label names match' in capsys.readouterr().out


def test_dataset_stage_refuses_a_permuted_label_mapping(tmp_path, monkeypatch):
    names = list(LABELS)
    names[0], names[15] = names[15], names[0]
    with pytest.raises(RuntimeError, match='differ from the notebook'):
        _dataset_stage(tmp_path, monkeypatch, names)
