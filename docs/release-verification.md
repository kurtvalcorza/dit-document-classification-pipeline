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
