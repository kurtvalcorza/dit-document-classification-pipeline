# Release verification

## Current decision

`tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb` remains **Candidate**. No hosted execution or real-model BYOD completion is claimed by this change. Checkpoint redistribution remains `pending-review` and hosting remains `HOLD`, as declared in the notebook; these statuses are not changed by local tests.

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
2. Runtime bootstrap still uses in-kernel installation. Public version matching avoids rejecting a CUDA local suffix, but does not prove the pin set resolves together. If packages were already imported, use **Runtime → Restart session** to preserve installed packages and then Run all. Record that run as restart-assisted; it is not evidence of uninterrupted fresh-runtime execution. Resolve and qualify that boundary before claiming that gate complete.
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
