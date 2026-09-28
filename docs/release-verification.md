# Release verification

## Current decision

`tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb` remains **Candidate** under DIMER Notebook Specification 2.2. Its default path has one successful hosted execution (Google Colab T4, 2026-09-26, recorded below) on notebook blob `56fb04d9`. The 2026-09-28 review fixes (below) change code cells, so that run no longer covers the current notebook: **Verification pending** until a hosted Run all of the fixed revision is recorded. The bring-your-own-pages branch (REL12) has no hosted execution yet. Checkpoint redistribution remains `pending-review`, as declared in the notebook, and local tests do not change it.

## Verification coverage (automatic versus manual)

| Check | How it runs | What it establishes |
|---|---|---|
| `ruff check src tests tools` | CI (`.github/workflows/ci.yml`), every pull request and push to `main` | lint only |
| `pytest` | CI | package validation, metrics and snapshot verification with synthetic fixtures; notebook setup, optional-path and name checks with model doubles; model-card and notebook-metadata conformance |
| `python tools/validate_release_assets.py` | CI | required files, model-card structure against Model Card Specification 1.2, notebook metadata against Notebook Specification 2.2, tutorials registry row, licence HOLD retained |
| Notebook `Run all` with the real model | manual, on a hosted GPU runtime | the default path end to end; recorded as a verification record in this file |
| Bring-your-own-pages branch with the real model | manual, not yet run | REL12 positive and negative BYOD evidence |

CI does not execute the notebook: the default path downloads a 343 MB checkpoint and a public dataset and is meant for a hosted GPU runtime. Static checks and unit tests are not execution evidence (Notebook Specification REL8).

## Local evidence — 2026-09-26

The unchanged baseline was reproduced before editing:

- `RUN_ROBUSTNESS=False` still called `batched_predict`.
- A ZIP with `a/page.png` and `b/page.png` silently retained only the second image.
- A second ZIP extracted into the shared directory retained the first run's unrelated pages.

The fixes change code cells 7, 31, 39 and 42. The 19 code cells parse, cell identities are retained, and the model/data/runtime pins are unchanged. There are no saved execution outputs. Guided explanations and infrastructure-collapse metadata are additional changes.

Run the focused offline checks without loading stale repository-wide test fixtures:

```shell
python -m pytest --noconftest tests/test_notebook_optional_paths.py -q
```

**Observed: 15 passed**, exit 0. These cases execute actual notebook functions/control paths with deterministic model doubles:

- public PEP 440 version matching accepts the exact release with a CUDA local suffix, but rejects mismatched, prerelease and postrelease versions;
- disabled robustness makes zero inference calls and supports a header-only export; enabled robustness runs all five variants;
- duplicate ZIP basenames are rejected and successive extractions have separate directories;
- labelled and unlabelled BYOD complete orchestration through JSON/CSV export with unique filename IDs, input digests and the correct measured/not-measurable verdict;
- duplicate, missing, extra or unknown labels fail before inference.

These tests demonstrate control flow and validation/export behavior. Model doubles do not establish model accuracy, GPU memory suitability, conversion parity or REL12 real-model completion. No dependencies or model weights were installed for this verification.

## Remaining exact-revision gates

1. Record the final commit and notebook Git blob, a fresh supported Colab runtime, device, package versions, clean-start/cache conditions, settings and all warnings. Execute default Run all with unchanged pins, including checkpoint digest verification, local SafeTensors conversion/parity, sample inference, metrics, robustness, galleries and exports. Retain the executed notebook and output digests.
2. Runtime bootstrap uses in-kernel installation and keeps a NumPy 2.x the hosted kernel has already imported. The 2026-09-26 Colab run completed in one `Run all` without a restart. If a future hosted image pre-imports another pinned package, the install cell stops with an instruction to use **Runtime → Restart session**; record any such run as restart-assisted, because it is not evidence of uninterrupted fresh-runtime execution (RUN10).
3. In a separate supported runtime, use authorized representative document-page images. Set `USE_BYOD=True`, `BYOD_PATH` to a staged ZIP/directory/image, and optionally `BYOD_LABELS_PATH` to a CSV with exactly one valid RVL-CDIP label per selected filename. Run the actual converted DiT model through validation, inference, metrics where labelled, and separate `byod_results.json`/`byod_predictions.csv` exports. Inspect identifiers, input digests, model revision and output counts. Keep sample and BYOD conclusions distinct.
4. In a separate negative run, supply either duplicate ZIP basenames or a labels CSV missing one image. Require a clear validation error before inference and retain the invalid input digest and rejection output. Also test an incompatible page image against the published shape limits.
5. Verify both `RUN_ROBUSTNESS` settings in the supported runtime. The disabled path should complete exports and terminal summary without transformed-model inference; the default enabled path must retain the original five-variant comparison. Keep the optional activity exploratory and do not tune on these previously inspected pages.

Do not promote the notebook from static parsing, model-double tests or runtime version checks alone. The release record must identify the exact executed revision and distinguish default, optional, restart-assisted and BYOD runs.


### Colab NumPy setup failure — 2026-09-26

The maintainer-supplied run stopped in setup before model execution: NumPy 2.1.3 was already loaded, while the notebook installed 2.5.3. The [failure record](execution-evidence/2026-09-26/colab-setup-failure.json) records the independently inspected error. The supplemental notebook now pins NumPy 2.1.3, preserving the observed Colab kernel version instead of replacing it. Other model/runtime pins are unchanged; stale-module detection remains enabled. Declared upstream requirements permit 2.1.3 (Transformers and datasets require >=1.17; the closed-set SciPy pin requires >=2.0,<2.8).

A regression executes the real setup prefix against a simulated Colab preloaded NumPy and package installer: it reproduces the original restart error before the fix and completes without a restart after it. This is setup regression evidence, not a full model/Colab rerun. A new hosted Run all is still required to discover any downstream issues. Use a fresh runtime for that rerun; the prior failed session already replaced installed packages.


### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb) and authorized merging PR #1 (merge commit `22b0686`). The file is archived byte-for-byte, SHA-256 `38a6253e8d60a0fc6532b72cbc2efc99aa70480692aebf8860ef8f3b7e1e01cc`. All 19 code cells have execution counts, 30 saved outputs and zero saved errors. Code-cell sources match commit `9e64856d38250b27ce63ff8b9eab2f311061c7b1`, tutorial blob `56fb04d914c2c7836446801ea27aeffd59b02a09`, apart from Colab-inserted `# @title` lines. Later commits on `main` that touch the notebook (`576dba4` (AI Use Disclosure)) change only markdown cells; its code cells are identical to the executed revision. This evidence commit does not change tutorial code.

Scope: Default path: 320 balanced RVL-CDIP-derived pages over 16 classes, checkpoint digest verification, local SafeTensors conversion and the five-variant robustness comparison. BYOD was not exercised.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, Transformers 4.57.6, datasets 4.1.1, huggingface_hub 0.36.2, NumPy 2.1.3, CUDA Tesla T4. Execution reaches the final completion summary. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Saved counts run sequentially from 1 to 19; runtime freshness and absence of manual restarts/reruns are not independently established by the artifact.

Results (sample-sanity measures on the built-in data, not general model rankings): Accuracy 0.9531, macro F1 0.9530, top-3 accuracy 0.9938 against random and majority baselines of 0.0625; conversion parity maximum absolute logit difference 0.0 (source checkpoint SHA-256 `1b7a901642c3…`, converted SafeTensors `b9484ea5054b…`). Robustness accuracy: original 0.812, rotate90 0.500, rotate180 0.688, low resolution 0.750, centre crop 0.812.

Status remains **Candidate**. Merge approval and this successful default-path run do not close the optional-path (FULL/BYOD) or REL12 qualification gates, and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored (editing it would change the verified blob).


## Review fixes — 2026-09-28

Source: *DiT Document-Type Classification Notebook — Review* of commit `a6e68f44a61f614fff1daaa4f55be0f2ae080bf3`, notebook blob `2c643bf6c780cbb64bfb3c7f5406b2e2bed8cd01` (verdict: Needs revision), with its offline probe bundle. `main` still pointed at the reviewed commit when the fixes started. The notebook has no generator; the fixes are anchored edits to existing cells, so no cell was added, removed or re-ordered and every cell ID is unchanged. Model, dataset and runtime pins are unchanged.

| Finding | Fix | Cells |
|---|---|---|
| DIT-M01 evidence computed but not shown | One page per class (gold labels only) is displayed before classification. A per-class table, weakest recall first, follows the principal metrics. Both galleries are displayed as well as saved. The robustness cell lists every page whose prediction changed and displays one deterministic original/transformed pair, or says that no page changed. BYOD prints a filename-to-prediction table and its output paths. | `18aafd5c`, `e3dd9263`, `3891d68d`, `100a15ce`, `907b911c`; prose `6141ed59`, `772e7bdb`, `3f050f2d`, `1d4d0e0d`, `guided-01` |
| DIT-M02 argmax versus argsort on ties | One helper, `rank_classes`, ranks by descending score with the lower class ID first on ties (column 0 equals `np.argmax`). Predictions, top-3, metrics, class-score ranks, robustness and BYOD all use it. The number of pages with a tied top-1 score is printed and exported. | `b485b2eb`, `a87b58d8`, `3891d68d`, `8be87936`, `907b911c`; prose `44f12e61`, `772e7bdb` |
| DIT-M03 multi-page TIFF silently reduced | `validate_byod_image` reads the frame count and header dimensions before decoding pixels, and rejects multi-frame or animated files with their page count. Phone-camera MPO JPEGs, whose extra frames are previews or depth maps, are classified by their primary image. | `907b911c`, prose `525cb60f` |
| DIT-m01 duplicate CSV headers | Duplicate column names and data rows with the wrong number of fields are rejected before the label mapping is built. | `907b911c`, prose `525cb60f` |
| DIT-m02 fixed-16-class macro F1 on BYOD | The definition is unchanged and now stated. Labelled BYOD prints and exports per-class support, predictions, recall and F1, and `classes_with_support`. A new `byod_class_scores.csv` exports all 16 scores and ranks per page. | `907b911c`, `d60ee8bb`, prose `525cb60f` |
| DIT-m03 provenance and wording | `provenance.json` records `notebook_spec` 2.2, the notebook repository and path, and the evaluation-subset limitations (official-test non-comparability, pre-training overlap, possible fine-tuning overlap). The conclusion prompt asks for the evaluation-subset result, not a held-out result. The dataset is loaded at the full reviewed commit `25b73ea8482e805ef40750464959e910e4579921` instead of resolving the current head and checking a prefix. | `8be87936`, `ca003003`, `guided-03`, metadata `dimer.dataset.revision` |
| DIT-m04 stale error gallery | An error-free run removes an earlier `high_confidence_errors.png`. The export cell lists exactly the files this run wrote, then any other files in the folder separately. | `100a15ce`, `8be87936`, prose `aeb01675` |

The packaged pipeline had the same tie defect (`np.argsort(-row)` in `predict`). It now ranks the logits with the same lower-class-ID rule through a pure-NumPy `rank_classes`, and `DECISION_RULE` states it.

**User-visible changes.** New rejections: multi-frame/animated BYOD images, labels CSVs with duplicate column names or wrong-width rows. New outputs: `byod_class_scores.csv`, a `correct` column in `byod_predictions.csv`, `per_class`, `classes_with_support` and `macro_f1_definition` in labelled `byod_results.json`, `pages_with_tied_top1_score` in `metrics.json`, the `notebook` and `evaluation_limitations` provenance fields. Changed values: `provenance.json` `notebook_spec` is `2.2`; the `input_manifest.json` decision rule states the tie rule; the notebook metadata key `dimer.dataset.revision_prefix` became `dimer.dataset.revision` (full SHA). The dataset cell no longer calls the Hub API to resolve the repository head.

### Offline verification (not clean-runtime evidence)

- **Before the fix, tie divergence under the notebook's pinned NumPy 2.1.3.** For a 16-class float32 vector with two tied maxima, `np.argsort(-v)[0]` disagreed with `np.argmax` for 28 of the 120 class pairs, including email/handwritten (2, 3). The same count was observed under NumPy 2.5.3. This shows the mechanism; it does not show that the recorded DiT run had ties.
- **Regression tests.** `tests/test_notebook_optional_paths.py` executes the notebook's own cells (by cell ID) with model doubles: the tie rule on unique, two-way, all-equal and rank-three ties; BYOD exports agreeing with accuracy on a tie; two-page TIFF rejected before inference, single-page TIFF and two-frame MPO phone JPEG accepted, oversized page rejected; malformed labels-CSV schemas rejected before inference; BYOD prediction and class-support printing; changed-page listing and the before/after display, with and without changes, and with robustness off; gallery display and stale-gallery removal; the pre-classification preview; provenance, wording and dataset-revision checks. Of the 23 new notebook tests, 20 fail on the reviewed notebook; the three that pass there are controls (single-page TIFF and MPO JPEG accepted, oversized page rejected). All 23 pass on the fixed notebook. `tests/test_pipeline.py` covers the package `rank_classes`.
- **CI-equivalent run** (Python 3.12, NumPy 2.5.3, Pillow 11.3.0, pytest 8.4.2, ruff 0.16.6, no torch): `ruff check src tests tools` clean; `pytest` 83 passed (baseline 54); `tools/validate_release_assets.py` passed. The same 83 tests pass under NumPy 2.1.3.
- **Ordered-cell harness.** Every code cell from the sample cell to the terminal summary ran in notebook order on 992 synthetic pages with a deterministic NumPy model double; installation, download, conversion, dataset and model-load cells were replaced by stubs. Journeys: default (31 pages with a tied top-1 score; accuracy recomputed from `predictions.csv` equals the reported accuracy; rank-1 in `class_scores.csv` equals every exported prediction; four displays); error-free rerun of the gallery and export cells (stale error gallery removed); robustness off (three displays, same sample accuracy); labelled BYOD ZIP; unlabelled single image (`not-measurable`); two-page TIFF (rejected). Results were identical under NumPy 2.1.3 and 2.5.3.

Synthetic pages and model doubles establish control flow, displays and exports only. They are not DiT inference, not conversion parity, and not execution on the pinned Colab stack.

### Remaining before promotion

1. A hosted Colab T4 Run all of the fixed revision with default settings, recorded with its commit, notebook blob, runtime and metrics. Compare the metrics with the 2026-09-26 run and report `pages_with_tied_top1_score`; any change in accuracy would come from ties and must be explained.
2. The same run, or a separate one, with `USE_BYOD=True`: a labelled ZIP or directory, a multi-page TIFF (expected rejection) and a malformed labels CSV (expected rejection), with the real converted model (REL12).
3. `RUN_ROBUSTNESS=False` in a hosted run.
4. The observed learner walkthrough that the review lists; it is outside what this repository can record automatically.

Status stays **Candidate**; redistribution stays `pending-review`.
