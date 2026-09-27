# DiT Document Classification Pipeline

DIMER pipeline for **Microsoft Document Image Transformer (DiT) Base fine-tuned on RVL-CDIP** (`microsoft/dit-base-finetuned-rvlcdip`). It classifies one rendered document page into the checkpoint's fixed 16-class RVL-CDIP vocabulary and returns the full 16-class score vector plus ranked predictions.

## Model

- Upstream: `microsoft/dit-base-finetuned-rvlcdip`
- Immutable revision: `23f8b03d130fb66bbc9a15df3c75d753e49240eb`
- Architecture: `BeitForImageClassification` / DiT Base
- Input: one RGB/grayscale-compatible rendered document-page image
- Output: 16 normalized model scores + top-1 document type
- Classes: letter, form, email, handwritten, advertisement, scientific report, scientific publication, specification, file folder, news article, budget, invoice, presentation, questionnaire, resume, memo

## Serving boundary

The upstream repository distributes `pytorch_model.bin` (343,362,393 bytes; SHA-256 `1b7a901642c36ec7e32de997683223faf02028624fb2e62a7e7e798a5dd1344e`). This repository does **not** serve that pickle. `tools/convert_weights.py` verifies the exact pinned source, statically audits its pickle globals, loads it with `torch.load(..., weights_only=True)`, converts the tensor state dict to `model.safetensors`, reloads it, checks tensor identity, and proves deterministic logit/prediction parity on a fixed probe before writing the DIMER manifest.

The serving runtime then loads only `config.json`, `preprocessor_config.json`, and `model.safetensors` from a digest-verified local snapshot with `trust_remote_code=False` and no network fallback.

## Quick start

```python
from PIL import Image
from dit_document_classification_pipeline import DiTDocumentClassificationPipeline

pipe = DiTDocumentClassificationPipeline.from_pretrained()
result = pipe.predict(Image.open("page.png"), top_k=3)
print(result["predictions"][0])
```

## Roles

- **Inference:** local, fixed 16-class RVL-CDIP classification.
- **Validator:** local image/batch/dimension/pixel-ceiling validation and labelled-dataset validation.
- **Evaluator:** accuracy, macro F1, top-3 accuracy, per-class precision/recall/F1, confusion matrix.
- **Finetuner:** intentionally not implemented in v0.1. Custom organizational document categories require a separate bounded adaptation contract rather than relabeling this fixed head.

## Weight conversion

```bash
python tools/convert_weights.py --download --remove-source
```

This produces a converted-only serving snapshot under `weights/dit-base-finetuned-rvlcdip/` and records the source identity, derived SafeTensors identity, tool versions, static pickle audit, and parity evidence in `dimer-base-manifest.json`.

## Tutorial notebook

[`tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb`](tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb) is a standalone `TASK-INFERENCE` / `WORKSHOP` notebook under DIMER Notebook Specification 2.2. It converts the pinned checkpoint to SafeTensors in the runtime, evaluates a deterministic 320-page RVL-CDIP-derived sample, and offers an optional bring-your-own-pages branch. Its registry entry and verification status are in [`tutorials/README.md`](tutorials/README.md) and [`docs/release-verification.md`](docs/release-verification.md). The model card ([`MODEL_CARD.md`](MODEL_CARD.md)) follows DIMER Model Card Specification 1.2.

## Licence

The code and notebooks in this repository are licensed under Apache-2.0 (`LICENSE`). The model weights are not redistributed here and are not covered by that licence; the upstream model repository declares no model-specific licence (see below).

## Release status

**HOLD.** The model-specific weight redistribution/licensing position is not explicit on the Hugging Face model repository, and Microsoft's public licensing clarification issue for DiT remains unresolved. The code can be reviewed and qualified, but DIMER public hosting must remain blocked until that decision is documented. See `STATUS.md` and `docs/WEIGHTS.md`.
