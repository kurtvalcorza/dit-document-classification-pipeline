---
license: other
license_name: not-declared-upstream
model_card_spec: "1.2"
pipeline_tag: image-classification
task: "Image Classification - Document Type"
base_model: microsoft/dit-base-finetuned-rvlcdip
date_published: "2022-03-07"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2022-03-07T20:48:42Z, https://huggingface.co/api/models/microsoft/dit-base-finetuned-rvlcdip); the DiT paper is arXiv:2203.02378 (2022-03); the pinned revision `23f8b03d130fb66bbc9a15df3c75d753e49240eb` is the Hub's `main` as last modified 2023-02-27"
---

# DiT Base fine-tuned on RVL-CDIP (`microsoft/dit-base-finetuned-rvlcdip` @ `23f8b03`) — Document-Type Classification

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2Fdit--base--finetuned--rvlcdip-ffcc4d?style=flat)](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-microsoft%2Funilm%2Fdit-181717?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/unilm/tree/master/dit)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2203.02378-b31b1b.svg)](https://arxiv.org/abs/2203.02378)
[![Model licence: not declared upstream](https://img.shields.io/badge/Model%20licence-not%20declared%20upstream-lightgrey.svg)](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** This repository does **not** redistribute the model weights: the notebook and the conversion tool download them from the upstream Hugging Face repository at the pinned revision. The upstream model repository declares no model-specific licence, so this repository does not establish the terms that govern your use of the weights, including any commercial use or redistribution (see *Licensing and redistribution* below). The accompanying code and notebooks are released under this repository's Apache-2.0 licence. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebook downloads third-party weights and datasets governed by their own terms and consumes compute on your own Colab or Kaggle account. To the maximum extent permitted by law, the maintainers of this repository accept no liability for any damages arising from their use. This repository implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides one ready-to-run notebook. It downloads the pinned upstream checkpoint, verifies its size and SHA-256, converts it to SafeTensors in the runtime and proves logit parity, classifies a deterministic balanced sample of 320 RVL-CDIP-derived test pages, compares the result with random and majority-class baselines, analyses errors and robustness, and writes machine-readable outputs with provenance:

- **Document-Type Classification Notebook** (`TASK-INFERENCE`, `WORKSHOP` mode):
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/dit-document-classification-pipeline/blob/main/tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb) [`DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb`](https://github.com/kurtvalcorza/dit-document-classification-pipeline/blob/main/tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb)
  *Whole-page classification into the checkpoint's fixed 16 RVL-CDIP classes, with accuracy, macro F1, top-3 accuracy, per-class metrics, a 16×16 confusion matrix, rotation/resolution/crop robustness, and an optional bring-your-own-pages branch that is disabled by default. The notebook performs no fine-tuning.*

---

#### Description

This repository packages `microsoft/dit-base-finetuned-rvlcdip` at the immutable Hugging Face revision `23f8b03d130fb66bbc9a15df3c75d753e49240eb`. The model is Microsoft's Document Image Transformer (DiT) Base. It is a Vision Transformer pre-trained with self-supervised masked image modelling on document images from IIT-CDIP, then fine-tuned by its authors for the 16-class RVL-CDIP document-type task (arXiv:2203.02378). It is architecture-compatible with BEiT: the pinned configuration builds `BeitForImageClassification` with 12 transformer layers, hidden size 768, 12 attention heads, 16×16 patches, 224×224 input and a 16-way classification head.

At inference, the pinned image processor resizes one page image to 224×224 and normalises it with mean and standard deviation 0.5. The transformer encodes the patch sequence, mean-pools it, and the head produces 16 logits. The predicted type is the `argmax` of those logits. No adaptation happens in this repository: there is no training, no in-context conditioning, and the class vocabulary is fixed by the upstream head.

This repository adds four things around the upstream weights. First, `tools/convert_weights.py` downloads the pinned `pytorch_model.bin`, checks its size and SHA-256, statically audits its pickle globals, loads it with `torch.load(..., weights_only=True)`, converts it to `model.safetensors`, and records tensor-level and logit-level parity in `dimer-base-manifest.json`. Second, the `DiTDocumentClassificationPipeline` class loads only that digest-verified SafeTensors snapshot, with `trust_remote_code=False` and no network fallback. Third, the package validates page images and labelled records before inference and computes classification metrics. Fourth, the tutorial notebook performs the same conversion, evaluation and inference inside a hosted notebook runtime.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is whole-page document-type classification. The input is one rendered page image, or a batch of 1–64 page images, each 32–4096 pixels per side and at most 32,000,000 pixels. The output of `DiTDocumentClassificationPipeline.predict` is the predicted class ID and label, the ranked top-*k* classes, and the 16 softmax scores (`class_scores`). The scores follow the fixed RVL-CDIP class order: letter, form, email, handwritten, advertisement, scientific report, scientific publication, specification, file folder, news article, budget, invoice, presentation, questionnaire, resume, memo.

The envisioned application domains are routing and triage of scanned office and archival correspondence into those 16 types, and teaching how a fixed-vocabulary image classifier behaves on document pages. The intended role in a larger system is a first-stage page router in a self-hosted document-processing application. It sends each page to a downstream step, such as OCR, form extraction, or document question answering, that this repository does not perform. It is also intended as a baseline against which a reader measures a model adapted to their own document taxonomy.

###### Primary Intended Users

The intended users are machine-learning engineers and data scientists who build document-processing applications, researchers who study document-image classification, and instructors and learners who use the notebook in teaching. The envisioned deployment settings are research, teaching, and self-hosted prototype applications that run the model on the user's own hardware or notebook runtime.

The pipeline assumes its users can render PDFs to page images themselves and can judge whether the 16 RVL-CDIP classes match their own document categories. It assumes they understand that softmax scores are not calibrated probabilities. It also assumes they can evaluate the model on labelled pages from their own collection before relying on it. The only measured accuracy comes from RVL-CDIP-derived pages that are close to the model's fine-tuning distribution.

###### Out-of-scope use cases

- **Capability boundaries.** The model assigns one of 16 fixed classes to one page. It does not read text (OCR), answer questions about a document, extract tables or fields, detect layout regions, aggregate predictions across the pages of a multi-page document, create new classes, or perform open-vocabulary classification. For document question answering, see the public sibling pipeline [`layoutlm-document-qa-pipeline`](https://github.com/kurtvalcorza/layoutlm-document-qa-pipeline); for table structure recognition, see [`table-transformer-structure-pipeline`](https://github.com/kurtvalcorza/table-transformer-structure-pipeline).
- **No reject class.** The model always returns one of the 16 classes. A page that belongs to none of them, such as a photograph, a receipt or a slide, still receives a label, often with a high score.
- **Input boundaries.** `validate_inputs` and `predict` reject images smaller than 32 pixels or larger than 4096 pixels on either side, images over 32,000,000 pixels, batches outside 1–64 images, and non-image inputs. PDFs are not accepted; render them page by page first.
- **Resolution loss.** Every page is resized to 224×224. Evidence carried by small print, stamps or signatures can be lost. The tutorial's low-resolution probe is one observation of this effect, not a characterisation of it.
- **Orientation.** The model is not rotation-invariant. In the recorded tutorial run, accuracy on the 16-page robustness probe fell from 0.812 on the original pages to 0.500 after a 90° rotation. Deskew and orient pages before classification.
- **Different taxonomies.** Relabelling the 16 outputs to an organisation's own categories is out of scope. This repository ships no fine-tuning entry point, and a relabelled head would not have been evaluated.
- **Decision boundaries.** Do not use the prediction to take an autonomous action with consequences for a person, such as rejecting an application, a claim or a filing, without human review of the page.

#### Factors

The model's behaviour varies with the kind of document, the capture process that produced the page image, and the distance between a user's pages and RVL-CDIP.

###### Groups

The pipeline is not human-centric: its unit is a document page, not a person, and its classes are document types. Group-level performance for people was not considered and is not measured.

Document pages nevertheless describe people and are written by people. Names, handwriting, languages and letterheads in a page can correlate with the demographic group of its author or subject. This repository did not audit the upstream IIT-CDIP pre-training corpus or the RVL-CDIP fine-tuning corpus for such group structure, and the upstream authors do not publish such an audit. A downstream operator who uses the model on pages that concern people is therefore expected to measure per-class accuracy separately for the language, region, script and handwriting subgroups present in their own collection. The operator should also check that errors do not concentrate on any one group before relying on the output.

###### Instrumentation

The training and evaluation pages were produced by document scanners. RVL-CDIP is a subset of the IIT-CDIP Test Collection, which consists of grayscale scans of documents released in US tobacco-industry litigation, stored as low-resolution TIFF images. The tutorial sample uses the public Hugging Face dataset `hf-tuner/rvl-cdip-document-classification`, a derived copy of RVL-CDIP. The notebook requires its revision to start with `25b73ea` and its test split to hold 992 rows.

The instrument characteristics that matter are resolution, bit depth, compression, skew and scanner noise. Every page is resized to 224×224 and converted to three-channel RGB, so detail below that resolution is not available to the model.

Instrument error reaches the model directly as image error. A changed scanner, a camera photo instead of a flatbed scan, heavy JPEG compression, rotation or cropping all shift the input distribution. The pipeline cannot detect these changes: it checks only image size and pixel count, not orientation, blur or compression. The tutorial's rotation, low-resolution and centre-crop probe shows the kind of shift involved; it is not a detector.

###### Environment

**Operating environment.** The package and notebook run on CPU or on a CUDA GPU. The notebook uses float32 throughout (`DTYPE = torch.float32`) and has been run on a Google Colab Tesla T4. No mixed precision, quantisation or compilation is used, and no feature requires particular hardware. The package requires Python 3.12 (`requires-python = ">=3.12,<3.13"`) and pins `torch==2.14.0`, `torchvision==0.29.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0` and `huggingface-hub==0.36.2`. The notebook pins the same model libraries plus `datasets==4.1.1`, and keeps the NumPy 2.x that a hosted kernel has already imported instead of reinstalling it.

**Data environment.** The model assumes the deployment pages resemble RVL-CDIP: scanned office documents from the late twentieth century, largely in English, in one of the 16 classes. Measured accuracy is only meaningful under that assumption, because the tutorial sample is drawn from the same distribution the checkpoint was fine-tuned on. The sample pages are IIT-CDIP images, which DiT saw during pre-training, and may include RVL-CDIP fine-tuning pages, so the measured accuracy may overstate performance on unseen documents. Expect degradation on born-digital renders, phone photographs, colour documents, non-English pages, modern document types such as receipts and slides, and pages rotated or cropped relative to a typical scan. No measurement in this repository covers those conditions.

#### Metrics

The measures below were chosen for a single-label, 16-class classifier whose output is used to route pages.

###### Performance Measures

`classification_metrics` and the notebook report these measures, named as the code names them:

- `accuracy` — the fraction of pages whose top-1 prediction equals the gold label. It is the discrete-correctness measure a router cares about first.
- `macro_f1` — the unweighted mean of the 16 per-class F1 scores. It gives every document type equal weight, so a model that fails on a rare class cannot hide behind the frequent ones.
- `top3_accuracy` — the fraction of pages whose gold label is among the three highest-scoring classes. It shows whether a human reviewer shown three suggestions would usually find the right one.
- `per_class` `precision`, `recall`, `f1` and `support`, and the 16×16 `confusion_matrix` — they show which types are confused with which, which a single summary number cannot.

The notebook also reports two baselines: the random-guess expectation (1/16 = 0.0625) and a majority-class predictor, which scores 0.0625 on the balanced sample. Reading `accuracy` alone would hide per-class failures; reading `macro_f1` alone would hide how often a single page is misrouted. None of these measures evaluates the calibration of the scores, which is not measured.

###### Decision thresholds

The pipeline's decision rule is `argmax(logits)` (`DECISION_RULE`); equal logits resolve to the lower class ID, and `top_k` uses the same order. This is an implicit threshold: the class with the highest score is predicted, however low that score is. No acceptance threshold, minimum score or reject option is applied, and none was set during development.

Thresholds are deliberately not shipped, because the cost of a misrouted page depends on the downstream step and the 16 scores are not calibrated. The deploying operator owns any threshold. The notebook's optional selective-classification experiment (`RUN_SELECTIVE_CLASSIFICATION`, off by default) shows the trade-off between the fraction of pages kept and their accuracy at margin thresholds from 0.0 to 0.50. An operator should set a margin or score threshold on their own labelled validation pages. Where a wrongly routed page is costly, for example a legal notice filed as an advertisement, choose a threshold that sends more pages to human review. Where review is costly and misrouting is cheap, a lower threshold is appropriate.

###### Approaches to uncertainty and variability

The notebook's reported metrics come from one evaluation of the pinned model on one deterministic sample. The sample holds 20 pages per class, 320 pages in total, drawn from the 992-row test split by ranking rows on the SHA-256 of `"{SAMPLE_SEED}:{row_index}:{label_id}"` with `SAMPLE_SEED = 42`. There is no repetition, cross-validation or resampling, so no standard deviation or confidence interval is reported. With 320 pages, one additional error changes accuracy by about 0.003; with 20 pages per class, one error changes a class's recall by 0.05.

Inference is deterministic up to floating-point differences between devices: there is no sampling, no dropout at inference, and no ensembling. The seed and the pinned dataset revision prefix fix the sample selection. Non-deterministic GPU kernels can change scores in the last digits and, for a page whose top two scores are nearly equal, the predicted class.

The `class_scores` are softmax-normalised logits. They are not calibrated probabilities that a prediction is correct, and the pipeline does not calibrate them. A caller who needs calibrated probabilities must fit a calibration method, such as temperature scaling, on labelled pages from their own distribution and evaluate it on a separate held-out set.

#### Ethical considerations and biases

No external ethics board or group-based testing has reviewed this model or pipeline, and this card does not imply that one did.

###### Data

The upstream model was pre-trained on IIT-CDIP, about 42 million document images, and fine-tuned on RVL-CDIP, 400,000 grayscale images in 16 classes drawn from IIT-CDIP (arXiv:2203.02378). IIT-CDIP comes from documents released in US tobacco-industry litigation. Those documents are real correspondence, forms and memos, and can carry names, addresses, signatures and other personal information of real people. The upstream authors do not publish an audit of personal or sensitive content, so its presence is not ruled out, and given the source it is likely.

This repository distributes code, the tutorial notebook, documentation, and one archived executed notebook under `docs/execution-evidence/`, whose saved outputs are text and tables. The repository does not distribute the model weights, the converted SafeTensors file, or RVL-CDIP images. The notebook downloads the weights and the sample at run time.

An operator who classifies their own pages is responsible for auditing them for personal, confidential or regulated information, and for deciding whether they may be processed in the chosen runtime. The pipeline performs no such audit, does no redaction, and does not detect sensitive content. The notebook's bring-your-own-pages branch warns against uploading such material unless the user is authorised to process it.

###### Human Life

No. The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life. Its only validation is the one execution recorded below, on RVL-CDIP-derived pages, by the maintainers of this repository. No independent, clinical, legal or regulatory validation has been performed.

Use in such domains is foreseeable, because pages routed in insurance claims, loan applications, hiring files or medical records fall into RVL-CDIP classes such as form, letter and resume. That use would be admissible only with human review of every page whose routing affects a person. It would also need an independent evaluation on the operator's own labelled pages, including the subgroups named under *Groups*, and any regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity.** `MODEL_REVISION` pins revision `23f8b03d130fb66bbc9a15df3c75d753e49240eb`. `tools/convert_weights.py` and the notebook require `pytorch_model.bin` to be exactly 343,362,393 bytes with SHA-256 `1b7a901642c36ec7e32de997683223faf02028624fb2e62a7e7e798a5dd1344e` before loading it. The conversion tool statically audits the pickle and refuses any global other than `collections.OrderedDict`, `torch.FloatStorage` and `torch._utils._rebuild_tensor_v2`, then loads it with `weights_only=True`.
- **Converted-only serving.** `verify_snapshot` refuses a snapshot unless its manifest names the pinned source, records `source.served` as `false`, binds `model.safetensors` to the source digest, and matches the size and SHA-256 of `config.json`, `preprocessor_config.json` and `model.safetensors`. It also requires identical predicted class IDs and a maximum absolute logit difference of at most `1e-6`. `from_pretrained` then loads with `local_files_only=True` and `trust_remote_code=False`, and rejects any state-dict key mismatch.
- **Label order.** The pipeline rejects a configuration whose `id2label` order differs from the canonical 16-class RVL-CDIP order, so scores cannot be exported under the wrong class names.
- **Input integrity.** `validate_inputs` and `predict` reject non-image inputs, batches outside 1–64, sides outside 32–4096 pixels, and images over 32,000,000 pixels, each with an error naming the violated limit. `validate_dataset` rejects duplicate or empty record IDs, records that do not hold exactly one image, and labels outside the 16 classes. The notebook's ZIP reader rejects absolute paths, `..` traversal, symlinks, duplicate basenames, archives over 512 MiB expanded, and archives with more than 200 images.
- **Statistical mitigations.** The tutorial sample is balanced at 20 pages per class, and the notebook rejects duplicate decoded pages. `macro_f1`, per-class metrics and the confusion matrix are reported alongside `accuracy`, so a failing class is visible.
- **Reproducibility.** `SAMPLE_SEED = 42` fixes the sample, and the notebook requires the dataset revision prefix `25b73ea` and a 992-row test split. Package and notebook dependencies are pinned. The notebook records runtime versions, the model revision and digests, and the dataset revision in `provenance.json`.
- **Refusals.** The pipeline ships no fine-tuning or relabelling entry point: `finetuner` is `not-implemented-v0.1` in `pipeline-manifest.json`. It never serves the upstream pickle.

###### Risks and harms

- **Confident error outside the training distribution.** The model always returns one of 16 labels, often with a high softmax score even for pages unlike RVL-CDIP. The operator and any person whose page is misrouted bear the harm. This is likely on born-digital, photographed, non-English or modern pages. The harm ranges from a delayed filing to a missed obligation, depending on the downstream step.
- **Orientation and resolution sensitivity.** A rotated, cropped or low-resolution page can be misclassified; the recorded probe dropped from 0.812 to 0.500 accuracy after a 90° rotation. This is realised whenever capture is not normalised, and the harm falls on the operator and the page's subject.
- **Amplification of corpus bias.** The model learned document conventions from US corporate documents of one industry and era. Pages that follow other conventions, including other languages, regions and handwriting styles, may be misrouted more often, which can systematically disadvantage the people who write or appear in those pages. This is unmeasured here and becomes more likely the further a collection is from RVL-CDIP.
- **Automation bias.** Reviewers may accept the top-1 label without inspecting the page, especially when the score is high. The operator and the page's subject bear the harm, which grows with the consequence of the routing decision.
- **Taxonomy mismatch.** An organisation's categories rarely match the 16 RVL-CDIP classes. Forcing pages into the nearest class misroutes them silently. This is realised in almost every deployment outside RVL-CDIP-like archives.
- **Data leakage.** Classifying pages in a hosted notebook sends them to that runtime provider, and exported CSV and JSON outputs name the input files and record their digests. The operator bears the harm of exposing personal or confidential pages, and its magnitude depends on the content.
- **Unclear weight licence.** The upstream repository declares no model-specific licence. A user who deploys or redistributes the weights may act outside the rights holder's terms. That harm falls on the user and is present in every commercial use.

###### Use cases

The following uses are unacceptable even where the model would work:

- surveillance of individuals or groups, for example profiling people by classifying their correspondence or personal files;
- unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, including automatic rejection of applications, claims or filings based on the predicted document type;
- deceptive or manipulative applications, such as presenting the output as a verified determination of a document's authenticity or legal category;
- any use that the upstream model terms or the deploying organisation's own terms prohibit. Because the upstream repository declares no model-specific licence, a user must establish those terms before commercial use or redistribution.

---

## Immutable provenance

| Item | Value |
|---|---|
| Upstream model | `microsoft/dit-base-finetuned-rvlcdip` |
| Pinned revision | `23f8b03d130fb66bbc9a15df3c75d753e49240eb` |
| Source file | `pytorch_model.bin`, 343,362,393 bytes, SHA-256 `1b7a901642c36ec7e32de997683223faf02028624fb2e62a7e7e798a5dd1344e` |
| Source serialization | PyTorch pickle; conversion input only, never served |
| Serving file | `model.safetensors`, written by `tools/convert_weights.py` or by the notebook |
| Converted SafeTensors SHA-256 (observed in the 2026-09-26 run) | `b9484ea5054bf02ede62493b4e7789a40873a513fb4a94c5aa786869d841d1f7` |
| Tutorial sample | `hf-tuner/rvl-cdip-document-classification`, revision prefix `25b73ea`, test split, 20 pages per class, `SAMPLE_SEED = 42` |

## Input/output contract

| Field | Contract |
|---|---|
| Input | one `PIL.Image.Image` or a sequence of 1–64; converted to RGB |
| Image size | 32–4096 pixels per side; at most 32,000,000 pixels |
| Preprocessing | pinned `AutoImageProcessor` (224×224, mean/std 0.5) |
| `predicted_class_id`, `predicted_label` | `argmax(logits)` over the fixed 16 classes; ties go to the lower class ID |
| `top_k` | ranked `{class_id, label, score}` entries; `top_k` in 1–16, default 3 |
| `class_scores` | 16 softmax scores in the canonical class order; not calibrated |
| `decision_rule`, `model_id`, `model_revision`, `device` | returned with every prediction batch |

## Licensing and redistribution

The Hugging Face model repository declares no model-specific licence. Microsoft's `unilm` source repository carries a repository-level licence, but that licence does not state terms for these weights, and a public question about commercial use of DiT on the `unilm` issue tracker has no resolution. This repository therefore does not redistribute the weights or the converted SafeTensors file, and does not infer permission to do so. The front-matter `license: other` records that no upstream licence is declared. The repository's own code is licensed under Apache-2.0 (`LICENSE`).

## Verification records

Executions of the tutorial notebook are recorded in [`docs/release-verification.md`](docs/release-verification.md). The one hosted execution so far:

- **Date:** 2026-09-26
- **Subject:** `tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb` at commit `9e64856d38250b27ce63ff8b9eab2f311061c7b1`, notebook blob `56fb04d9`; the executed copy is archived under `docs/execution-evidence/2026-09-26/`
- **Runtime:** Google Colab, Tesla T4, Python 3.13.15, `torch 2.14.0+cu130`, `transformers 4.57.6`, `datasets 4.1.1`, `huggingface_hub 0.36.2`, NumPy 2.1.3
- **Procedure:** `Run all` with default settings; 19 code cells with sequential execution counts; the bring-your-own-pages branch disabled
- **Observed result:** on the 320-page sample, `accuracy` 0.9531, `macro_f1` 0.9530 and `top3_accuracy` 0.9938, against 0.0625 for the random and majority-class baselines; conversion parity maximum absolute logit difference 0.0; robustness probe accuracy on 16 pages: original 0.812, rotate90 0.500, rotate180 0.688, low resolution 0.750, centre crop 0.812
- **Caveats:** one run on one sample drawn from the checkpoint's own fine-tuning distribution, so it is sample-sanity evidence, not a benchmark or evidence of transfer; the exported files were not inspected separately; the artifact does not establish that the runtime was fresh; the bring-your-own-pages branch was not exercised

## References

- Li, J., Xu, Y., Lv, T., Cui, L., Zhang, C., Wei, F. *DiT: Self-supervised Pre-training for Document Image Transformer.* arXiv:2203.02378, 2022. <https://arxiv.org/abs/2203.02378>
- Harley, A. W., Ufkes, A., Derpanis, K. G. *Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval.* ICDAR 2015 (RVL-CDIP). <https://arxiv.org/abs/1502.07058>
- Upstream model: <https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip>
- Upstream source: <https://github.com/microsoft/unilm/tree/master/dit>
