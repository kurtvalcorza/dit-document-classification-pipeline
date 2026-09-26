# Document-type classification notebook

| Notebook | Profile | Mode | Carrier | Default runtime | Sample | BYOD | Run all | Release status |
|---|---|---|---|---|---|---|---|---|
| [Document-Type Classification with DiT](DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb) | `TASK-INFERENCE` | `WORKSHOP` | standalone | CPU or CUDA GPU | deterministic balanced 320-page RVL-CDIP-derived subset, 20 per class | image, directory or ZIP; optional complete filename/label CSV; disabled by default | exact-revision hosted qualification pending; bootstrap may require Restart session | **Candidate** |

The notebook applies a fixed 16-class classifier to whole document-page images. It does not perform OCR, document QA or model adaptation. Model/data revisions and runtime pins are declared inside the notebook; the task contract is in [the workshop specification](../docs/document-type-classification-workshop-spec.md).

The guided layer asks for predictions before classification and robustness comparisons, provides an interpretation checkpoint, and separates concept/evaluation work from collapsed infrastructure. The optional activity compares existing original/rotation/resolution results on identical pages without changing canonical predictions or tuning on the inspected test sample.

`RUN_ROBUSTNESS=False` skips transformed-model inference while retaining a header-only `robustness.csv` and an empty summary. BYOD ZIP inputs reject duplicate basenames and use a unique extraction directory per attempt. A supplied labels CSV must match the selected filenames exactly; absent labels produce prediction-only output. BYOD JSON/CSV exports are separate from sample reports and record input digests and model provenance.

Edit this notebook directly: this branch has no tracked notebook generator. Preserve its model/data pins and Candidate status. Local optional-path checks are in `tests/test_notebook_optional_paths.py`; they use model doubles and do not establish real-model or hosted execution. See [release verification](../docs/release-verification.md) for measured checks and remaining gates.
