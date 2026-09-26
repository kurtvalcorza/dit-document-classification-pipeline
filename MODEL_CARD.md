---
model_card_spec: "1.2"
pipeline_tag: image-classification
base_model: microsoft/dit-base-finetuned-rvlcdip
license: other
---

# DiT Base — RVL-CDIP Document-Type Classification

## Description

This pipeline packages Microsoft's `microsoft/dit-base-finetuned-rvlcdip`, a Document Image Transformer (DiT) base model pre-trained on IIT-CDIP and fine-tuned for the 16-class RVL-CDIP document-image classification task. DiT is architecture-compatible with BEiT; the pinned Hugging Face configuration uses `BeitForImageClassification` with 12 transformer layers, hidden size 768, 12 attention heads, 16×16 patches, 224×224 inputs, mean pooling, and 16 labels.

## Intended use

The pipeline classifies a **single rendered document page** into one of the fixed RVL-CDIP document types. It is appropriate for research, evaluation, workshop demonstration, and self-hosted document-routing prototypes where those 16 classes are meaningful.

It does not perform OCR, document question answering, table extraction, layout detection, multi-page aggregation, open-vocabulary classification, or class creation. PDFs must be rendered page-by-page before use.

## Output semantics

The reported class is `argmax(logits)`. `class_scores` are softmax-normalized model scores over the fixed 16-class vocabulary; they are not calibrated probabilities of real-world correctness. The model has no reject/unknown class.

## Input boundary

Images are converted to RGB. Each side must be 32–4096 pixels and each image must contain at most 32 million pixels. The pinned processor resizes to 224 px and normalizes with mean/std 0.5, so small text and fine visual evidence can be lost.

## Evaluation

`evaluate()` reports accuracy, macro F1, top-3 accuracy, per-class precision/recall/F1, and a 16×16 confusion matrix for labelled RVL-CDIP-compatible page images. Measurements on RVL-CDIP are in-distribution/task-aligned evidence because the checkpoint itself was fine-tuned on RVL-CDIP; they are not evidence of transfer to a different organization's document taxonomy.

## Supply chain

The upstream `pytorch_model.bin` is pinned at revision `23f8b03d130fb66bbc9a15df3c75d753e49240eb`, 343,362,393 bytes, SHA-256 `1b7a901642c36ec7e32de997683223faf02028624fb2e62a7e7e798a5dd1344e`. It is never a serving artifact. The maintainer conversion tool statically audits the pinned pickle globals, loads it only after digest verification with `weights_only=True`, converts the tensor state dict to SafeTensors, reloads it, checks tensor names/shapes/dtypes/values, and records logit and predicted-class parity. Runtime loading is local-only with `trust_remote_code=False`.

## Licensing / redistribution

The Hugging Face model repository does not expose an explicit model-specific license tag. Microsoft's `unilm` source repository is licensed at repository level, but a public DiT commercial-licensing clarification issue remains unresolved. This repository therefore records `license review required`, `redistribution pending-review`, and `hosting HOLD`; it does not infer weight-redistribution clearance from the repository-level license.

## Privacy

Document pages may contain personal, financial, employee, proprietary, or regulated information. Operators should not submit such material to hosted notebook environments unless authorized. This package itself performs local inference and does not call a DIMER service.
