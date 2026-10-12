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

def test_confusion_guidance_names_label_noise_counts_and_source():
    text = cell('601e66d3')
    assert 'out of 20' in text and '1 or 2 pages is an anecdote' in text
    assert '8.1%' in text and 'quoted from the paper, not measured in this notebook' in text
    assert 'arXiv:2306.12550' in text and 'Larson, Lim and Leach (2023)' in text
    assert 'model error, a label error, or a page that fits two classes' in text
    assert 'out of 20 pages' in cell('52854116')


def test_error_gallery_activity_asks_for_a_label_error_judgement():
    text = cell('1d4d0e0d')
    assert '### Activity: model error, label error, or ambiguous page?' in text
    assert '**label error**' in text and '<details><summary>Sample answer</summary>' in text
    conclusion = cell('guided-03')
    assert 'count out of 20 pages' in conclusion and 'gold labels of those pages looked right' in conclusion


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
    assert meta['review_revisions'][-1]['change'] == 'review round 2026-10-12 (DIT2-M1, DIT2-m1..m6)'
    assert 'pending' not in meta['review_revisions'][0]['hosted_execution']
    opening = cell('11007f70')
    assert '**Notebook kind:** `WORKSHOP`' in opening
    notice = opening.index('> **⚠️ Notice.** For research, training and evaluation only.')
    assert notice < opening.index('By the end you will be able to:')
    assert 'Data you bring stays in your runtime' in opening and 'accept no liability' in opening
