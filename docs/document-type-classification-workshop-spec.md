# DIMER Document-Type Classification Workshop

## RVL-CDIP with Microsoft Document Image Transformer (DiT)

**Proposed filename:** `DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb`  
**Notebook Specification:** DIMER `NOTEBOOK_SPEC.md` **v2.1**  
**Profile:** `TASK-INFERENCE`  
**Pedagogical mode:** `WORKSHOP`  
**Standalone:** `true`  
**Canonical workflow:** Frozen document-image classification and robustness analysis  
**Adaptation:** None in this notebook  
**Recommended runtime:** CPU or CUDA GPU  
**Task:** Whole-page document-type classification  
**Model:** Microsoft DiT-base fine-tuned on RVL-CDIP  
**Classes:** 16  
**Input unit:** One rendered document page image  
**Output:** One 16-class score vector + top-1 document type

---

# 1. Purpose

This notebook teaches **whole-page document-type classification**:

```text
document page image
        ↓
Document Image Transformer
        ↓
16 document-type scores
        ↓
predicted document class
```

The notebook uses Microsoft DiT-base fine-tuned on RVL-CDIP, a document-domain vision transformer pretrained on IIT-CDIP and subsequently fine-tuned on the 16-way RVL-CDIP document classification task.

The workshop focuses on:

- document-image classification;
- document-domain visual representations;
- whole-page layout cues;
- fixed class vocabularies;
- class-score interpretation;
- confusion between visually similar document types;
- page orientation;
- page cropping;
- resolution degradation;
- confidence versus correctness;
- class-level error analysis; and
- applying a fixed RVL-CDIP classifier to user document-page images.

This notebook does **not** perform OCR, question answering, table extraction, layout detection, or generative document understanding.

---

# 2. Why this deserves its own notebook

Document classification answers:

```text
What kind of document is this page?
```

Document QA answers:

```text
What information does this document contain?
```

OCR/document extraction answers:

```text
What text or structured content is present?
```

These are different tasks with different evaluation contracts.

This workshop therefore remains separate from:

- LayoutLM Document QA;
- Pix2Struct DocVQA;
- SmolDocling;
- GOT-OCR;
- Table Transformer;
- DePlot; and
- future multimodal document-generation workflows.

---

# 3. Learning objectives

By the end of the notebook, the learner should be able to:

1. explain document-type classification as whole-page image classification;
2. describe why document-domain pretraining differs from ordinary natural-image pretraining;
3. classify pages into the 16 RVL-CDIP categories;
4. inspect the complete 16-class score vector rather than only the top prediction;
5. compute accuracy, macro F1, top-3 accuracy, per-class precision and recall;
6. read and interpret a document-type confusion matrix;
7. identify common document-type confusions;
8. distinguish model score from calibrated probability;
9. measure the effect of page rotation;
10. measure the effect of cropping and low resolution;
11. understand the single-page boundary of the classifier; and
12. classify compatible user-supplied page images locally.

---

# 4. Canonical model

## 4.1 Model identity

**Model:** `microsoft/dit-base-finetuned-rvlcdip`

**Immutable Hub revision:**

```text
23f8b03d130fb66bbc9a15df3c75d753e49240eb
```

This is the current immutable repository head represented by the latest recorded commit.

The model is an image classifier implemented as `BeitForImageClassification`. Its configuration specifies:

- 12 transformer layers;
- hidden size 768;
- 12 attention heads;
- 16×16 image patches;
- 224×224 model input;
- mean pooling;
- 16 output labels.

---

# 5. Why DiT

DiT is specifically pretrained on large-scale document imagery rather than ordinary natural-image classification data. The upstream project describes DiT as a self-supervised Document Image Transformer pretrained on IIT-CDIP and intended for tasks such as document image classification, table detection, and layout analysis.

The official DiT project reports a fine-tuned RVL-CDIP result of 92.11% for the base model. That value is **upstream evidence only** and MUST NOT be presented as a result reproduced by this notebook.

---

# 6. Canonical class ordering

The class ordering MUST match the model configuration exactly:

| ID | Document type |
|---:|---|
| 0 | letter |
| 1 | form |
| 2 | email |
| 3 | handwritten |
| 4 | advertisement |
| 5 | scientific report |
| 6 | scientific publication |
| 7 | specification |
| 8 | file folder |
| 9 | news article |
| 10 | budget |
| 11 | invoice |
| 12 | presentation |
| 13 | questionnaire |
| 14 | resume |
| 15 | memo |

This ordering is load-bearing and must be preserved in score exports.

---

# 7. Model input contract

The public pipeline accepts:

```text
one RGB or grayscale-compatible document-page image
```

The image is converted to RGB before processing.

Recommended DIMER ceilings:

```text
MIN_IMAGE_SIDE = 32
MAX_IMAGE_SIDE = 4096
MAX_PIXELS = 32_000_000
```

One page is one classification unit.

A PDF is **not** directly classified.

A PDF must first be rendered page-by-page outside this pipeline.

---

# 8. Model preprocessing

The pinned DiT processor specifies:

- resize enabled;
- target size 224;
- center crop disabled;
- normalization enabled;
- mean `[0.5, 0.5, 0.5]`;
- standard deviation `[0.5, 0.5, 0.5]`.

The notebook must use the exact carried processor configuration.

No notebook-specific preprocessing should silently replace it.

The learner-facing explanation should emphasize:

> A page that may originally contain millions of pixels is ultimately represented at 224×224 resolution.

That compression is important when interpreting behavior on small text, dense tables, or visually similar page layouts.

---

# 9. Output contract

For each page:

```text
image_id
predicted_class_id
predicted_label
top1_score
top3
class_scores[16]
model_id
model_revision
```

The decision rule is:

```text
predicted class = argmax(logits)
```

The exported class scores are:

```text
softmax(logits)
```

Softmax scores MUST be described as:

> normalized model scores over the fixed 16-class label set, not calibrated probabilities of real-world correctness.

---

# 10. Model asset issue

The official Hugging Face repository currently distributes the principal weight file as:

```text
pytorch_model.bin
```

with size approximately 343 MB. Hugging Face identifies it as pickle-based PyTorch serialization and shows detected pickle imports.

DIMER MUST NOT serve this file directly.

---

# 11. Required DIMER conversion

Before this notebook can become release-grade, perform a one-time trusted conversion:

```text
official pytorch_model.bin
        ↓
static/security audit
        ↓
torch.load(..., weights_only=True)
        ↓
state_dict validation
        ↓
safetensors conversion
        ↓
model.safetensors
```

The DIMER snapshot MUST contain the converted SafeTensors representation.

The upstream `.bin` is source provenance only and MUST NOT be part of the serving snapshot.

---

# 12. Conversion provenance

The model asset manifest must record:

```text
canonical model ID
immutable Hub revision

source:
  pytorch_model.bin
  source bytes
  source SHA-256
  serialization = PyTorch pickle

conversion:
  tool
  tool version
  torch version
  safetensors version

derived:
  model.safetensors
  derived bytes
  derived SHA-256
  derived_from_sha256
```

The conversion must satisfy DIMER `MODEL_ASSET_SPEC` derived-asset requirements.

---

# 13. Conversion parity

Qualification MUST compare the source and converted model on a fixed set of document images.

Required checks:

```text
all state_dict tensor names identical
all tensor shapes identical
all tensor dtypes identical

converted model loads successfully

predicted class IDs identical

max absolute logit difference <= explicit tolerance
```

Recommended tolerance:

```text
max_abs_logit_diff <= 1e-6
```

for an unmodified float32 state-dict conversion.

If exact-equivalent output cannot be demonstrated, qualification fails.

---

# 14. License and redistribution gate

The Hugging Face model page does not currently provide a model-specific license declaration, while the upstream `microsoft/unilm` project points to the license at its repository root. A public DiT licensing clarification issue remains open.

Therefore the initial DIMER asset state should be:

```text
license_review = required
redistribution_status = pending-review
hosting_status = HOLD
```

Do not infer DIMER redistribution clearance solely from the repository-level license.

The technical notebook specification can proceed, but public DIMER hosting must remain blocked until the model-weight redistribution decision is documented.

---

# 15. Dataset

Use **RVL-CDIP**, the benchmark on which the checkpoint was fine-tuned.

The canonical RVL-CDIP dataset contains:

- 400,000 grayscale document images;
- 16 classes;
- 25,000 images per class;
- 320,000 training images;
- 40,000 validation images;
- 40,000 test images.

The images derive from IIT-CDIP / the Legacy Tobacco Document Library lineage, and the Hugging Face dataset card uses the license tag `other`.

The notebook must therefore identify RVL-CDIP's data provenance and must not imply that the dataset has a simple permissive software-style license.

---

# 16. Canonical tutorial evaluation sample

Do **not** download the full approximately 38.8 GB dataset during `Run all`.

Instead create a pinned balanced tutorial sample from the **official RVL-CDIP test split**:

```text
20 pages × 16 classes = 320 pages
```

This sample is large enough to expose real class confusions while remaining practical for a notebook.

---

# 17. Sample selection rule

At build time:

1. resolve the official RVL-CDIP test split at an immutable dataset revision;
2. group records by the canonical 16 labels;
3. generate a deterministic ranking key:

```text
SHA256("42:" + stable_source_key)
```

4. sort within each class by the ranking key;
5. select the first 20 records per class.

No model predictions may be inspected during selection.

---

# 18. Sample manifest

Generate:

```text
rvlcdip_sample_manifest.json
```

containing for every selected page:

```text
sample_id
source_split
source_row_key
class_id
class_label
width
height
source_bytes
image_sha256
```

Global fields:

```text
dataset_id
dataset_revision
sample_seed = 42
images = 320
images_per_class = 20
sample_digest
```

The sample digest should be deterministic over:

```text
source key
label
decoded image digest
```

---

# 19. Dataset acquisition release gate

Because RVL-CDIP's dataset license metadata is not a simple permissive license, the notebook repository SHOULD NOT embed the 320 page images directly by default.

Preferred acquisition order:

```text
1. fetch selected sample rows from the pinned upstream dataset
2. verify image SHA-256
3. decode locally
```

If efficient row-level upstream access is unavailable, a DIMER-controlled tutorial sample mirror may be used only after separate dataset redistribution review.

The notebook MUST NOT fall back to downloading the entire dataset silently.

---

# 20. Evaluation-sample integrity

Before inference assert:

```text
320 images present
16 classes present
20 images per class

all image digests match
all class IDs are 0..15
all labels match model id2label
all images decode successfully
no duplicate decoded pixels
```

The notebook should print a compact class-count table.

---

# 21. Dataset interpretation

The sample is:

```text
sample_kind = "balanced RVL-CDIP official-test tutorial subset"
```

It is held out from the standard RVL-CDIP training split.

However, because the checkpoint itself was fine-tuned on RVL-CDIP, this notebook measures **task-aligned held-out RVL-CDIP performance**, not transfer to an unseen document domain.

---

# 22. Runtime

Use the standard current DIMER Python 3.12 runtime family:

```text
torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0
transformers==4.57.6
safetensors==0.8.0
numpy==2.5.3
pillow==11.3.0
huggingface-hub==0.36.2
```

The notebook must print:

```text
Python
PyTorch
TorchVision
Transformers
SafeTensors
NumPy
Pillow
device
dtype
CUDA device if present
```

CPU should be supported.

CUDA may be used automatically where available.

---

# 23. Model loading

Load only the verified converted DIMER snapshot:

```text
weights/dit-base-finetuned-rvlcdip/
```

Required files:

```text
config.json
preprocessor_config.json
model.safetensors
README / provenance notice as permitted
dimer-base-manifest.json
```

Loader:

```text
AutoImageProcessor
AutoModelForImageClassification
```

with:

```text
local_files_only=True
trust_remote_code=False
```

No network fallback is permitted after snapshot verification.

---

# 24. Baselines

The evaluation should include:

## Uniform random expectation

With 16 balanced classes:

```text
accuracy expectation = 1 / 16 = 6.25%
```

## Majority baseline

Compute the majority class from the evaluation labels.

Because the tutorial sample is exactly balanced, all classes tie.

Use deterministic tie-breaking by lowest class ID:

```text
letter
```

Do not claim the majority baseline represents a useful document classifier; it exists as a classification floor.

---

# 25. Principal metrics

Report:

### Accuracy

Fraction of pages with correct top-1 prediction.

### Macro F1

Equal-weight average F1 over all 16 document classes.

This prevents classes with more successful predictions from dominating interpretation.

### Top-3 accuracy

Fraction where the gold document type appears among the three highest model scores.

This is useful because several RVL-CDIP categories are visually related.

---

# 26. Per-class metrics

For every class report:

```text
support
precision
recall
F1
top-3 recall
```

Do not report only aggregate accuracy.

---

# 27. Confusion matrix

Produce both:

```text
raw confusion counts
row-normalized confusion matrix
```

Canonical class ordering MUST remain identical to the model config.

The matrix should identify the strongest off-diagonal confusion pairs automatically.

---

# 28. Error analysis

Automatically identify:

### High-confidence incorrect

Incorrect predictions with the highest top-1 model scores.

### Low-margin predictions

Pages where:

```text
top1_score - top2_score
```

is smallest.

### Correct but uncertain

Correct pages with the smallest top-1/top-2 margin.

### Strong confusion pairs

The largest off-diagonal confusion counts.

For each category, display several deterministic examples.

---

# 29. Workshop prediction exercise

Before showing the confusion matrix, ask the learner to predict which classes may be visually difficult to distinguish.

Useful pairs to consider include:

```text
scientific report ↔ scientific publication
letter ↔ memo
form ↔ questionnaire
email ↔ memo
budget ↔ invoice
```

These are hypotheses only.

The notebook must reveal the actual measured confusion matrix rather than assert these pairs must dominate.

The exercise is non-blocking.

---

# 30. Top-3 interpretation

For selected examples show:

```text
gold label

1. predicted class     score
2. second class        score
3. third class         score
```

This helps distinguish:

```text
clear error
```

from:

```text
semantically / visually ambiguous classification
```

The notebook should not reinterpret the ground-truth class based on model alternatives.

---

# 31. Robustness experiment

Document-page classification is particularly sensitive to scan/render conditions.

Use a predetermined probe set:

```text
16 images
1 per class
```

Select the lowest sample-manifest ranking key within each class.

The robustness experiment runs the same page under multiple controlled transformations.

---

# 32. Orientation variants

Evaluate:

```text
original
rotate 90° clockwise
rotate 180°
```

The true document class remains unchanged.

For every transform report:

```text
top-1 accuracy
prediction stability
mean gold-class score
mean top1 margin
```

Prediction stability means:

```text
fraction whose predicted class equals the original-image prediction
```

---

# 33. Resolution degradation

Create:

```text
low-resolution
```

by reducing the page to approximately 112×112 before upsampling it again for the normal processor.

The purpose is to deliberately erase fine visual/text detail while retaining coarse layout.

Measure:

```text
accuracy
prediction stability
gold-score change
```

This is a robustness probe, not a production preprocessing recommendation.

---

# 34. Crop sensitivity

Create a deterministic center crop retaining approximately:

```text
80% of page width
80% of page height
```

then feed it through the normal processor.

This tests sensitivity to:

- missing headers;
- missing footers;
- truncated forms;
- changed page layout.

Report the same stability metrics.

---

# 35. Robustness output

Produce:

| Variant | Accuracy | Stability vs original | Mean gold score | Mean margin |
|---|---:|---:|---:|---:|
| original | measured | 1.0 | measured | measured |
| rotate90 | measured | measured | measured | measured |
| rotate180 | measured | measured | measured | measured |
| low_resolution | measured | measured | measured | measured |
| center_crop | measured | measured | measured | measured |

No robustness threshold should be treated as a release requirement.

---

# 36. Confidence semantics

For every page compute:

```text
top1_score
top2_score
margin = top1 - top2
entropy
```

These are uncertainty diagnostics.

The notebook MUST state:

> Softmax scores and margins are model outputs, not calibrated real-world correctness probabilities.

No arbitrary reject threshold should be introduced.

---

# 37. Optional selective-classification experiment

A small optional section may examine:

```text
margin >= threshold
```

or:

```text
top1_score >= threshold
```

against accuracy/coverage.

This is exploratory only.

It must not claim a universal abstention threshold.

Default:

```text
RUN_SELECTIVE_CLASSIFICATION = False
```

---

# 38. Visualization

Create a 4×4 document-class gallery containing one deterministic example from each class.

Each tile should show:

```text
gold class
predicted class
top1 score
```

A second panel should display selected misclassifications.

Visualizations supplement machine-readable exports.

---

# 39. BYOD

Default:

```text
USE_BYOD = False
```

Accept either:

```text
one page image
```

or:

```text
directory / ZIP of page images
```

Optional labels file:

```text
labels.csv
```

with:

```text
filename,label
```

---

# 40. BYOD limits

Recommended workshop limits:

```text
1..200 page images
side length 32..4096 px
<=32 megapixels per image
```

Supported labels, when provided, must be exactly one of the 16 RVL-CDIP classes.

If labels are absent:

```text
evaluation verdict = not-measurable
```

but predictions and class scores are still exported.

If labels are present:

```text
accuracy
macro F1
top-3 accuracy
per-class metrics
```

may be reported.

---

# 41. BYOD class boundary

This checkpoint can classify only the fixed RVL-CDIP vocabulary.

For example, a user cannot add:

```text
purchase order
passport
medical certificategovernment memorandum
tax return
```

as new classes merely by changing a label list.

New document categories require re-heading / fine-tuning.

That belongs in a separate future notebook:

**Custom Document-Type Adaptation with DiT**

---

# 42. BYOD privacy

Document images may contain:

- names;
- addresses;
- signatures;
- financial data;
- employee information;
- proprietary records;
- personal correspondence.

The notebook must warn users not to upload confidential, restricted, personal, or regulated documents to a hosted notebook environment unless authorized.

User images remain in the notebook runtime and are not sent to a DIMER inference service.

---

# 43. Required outputs

Use:

```text
outputs/document_type_classification/
```

---

# 44. `predictions.csv`

One row per page:

```text
image_id
true_label
predicted_class_id
predicted_label
correct
top1_score
top2_label
top2_score
top3_label
top3_score
margin
entropy
```

`true_label` may be empty for unlabelled BYOD.

---

# 45. `class_scores.csv`

Long-format scores:

```text
image_id
class_id
class_label
score
rank
```

Exactly 16 rows per input page.

This explicitly preserves class ordering.

---

# 46. `class_metrics.csv`

```text
class_id
class_label
support
precision
recall
f1
top3_recall
```

---

# 47. `confusion_matrix.csv`

Matrix with:

- rows = gold classes;
- columns = predicted classes;
- canonical ordering fixed to IDs 0–15.

---

# 48. `robustness.csv`

```text
image_id
gold_label
variant
predicted_label
correct
top1_score
gold_score
margin
same_as_original_prediction
```

---

# 49. `metrics.json`

Contains:

```text
sample identity
majority baseline
random expectation
accuracy
macro F1
top-3 accuracy
per-class metrics
confusion matrix
error-analysis summaries
robustness summaries
timings
```

---

# 50. `provenance.json`

Must record:

```text
notebook_spec
profile
pedagogical_mode

model ID
immutable model revision

source pytorch_model.bin:
  size
  SHA-256

converted model.safetensors:
  size
  SHA-256
  derived_from

conversion tool versions
conversion parity result

config digest
processor digest

class ordering

RVL-CDIP:
  dataset ID
  immutable revision
  source split
  sample seed
  sample selection rule
  sample size
  image digests
  sample digest

runtime versions
device
dtype
timings
```

No secrets.

---

# 51. Input manifest

Export:

```text
input_manifest.json
```

including:

```text
schema
image count
dimension ceilings
observed dimensions
labels if present
validation verdict
findings
```

Include at least one non-blocking rejection probe, such as an image smaller than the minimum dimension.

---

# 52. Runtime timing

Record separately:

```text
snapshot verification time
model load time
warm-up inference
per-page inference times
total sample inference time
```

Report:

```text
mean
median
minimum
maximum
```

GPU timing must synchronize CUDA.

Runtime measurements apply only to the recorded environment.

---

# 53. No fine-tuning in this notebook

This is deliberate.

The checkpoint is already fine-tuned on RVL-CDIP. Fine-tuning it again on a small RVL-CDIP tutorial subset would blur the distinction between:

```text
evaluating the existing document classifier
```

and:

```text
training a new classifier
```

Therefore this notebook is:

```text
TASK-INFERENCE
```

not:

```text
E2E
```

---

# 54. Future E2E notebook

A later notebook may support:

**Custom Document-Type Adaptation with DiT**

using:

```text
document-domain DiT backbone
→ reinitialize classifier head
→ user-defined class vocabulary
→ frozen linear probe
→ bounded last-block unfreeze
→ validation selection
→ held-out evaluation
→ SafeTensors adapter export
→ fresh reload
```

That is separate from the RVL-CDIP workshop.

---

# 55. Explicit non-goals

This notebook does not perform:

- OCR;
- text extraction;
- PDF rendering;
- multi-page aggregation;
- document QA;
- form-field extraction;
- layout detection;
- table detection;
- table QA;
- semantic search;
- document embeddings as a primary output;
- generative classification;
- fine-tuning;
- class creation;
- open-vocabulary classification;
- DIMER API calls.

---

# 56. Standalone implementation

The notebook MUST NOT:

```text
git clone
pip install -e .
import a repository-local DIMER package
download DIMER Python source
call a DIMER worker/service
```

The notebook carries its own:

```text
snapshot verifier
classification wrapper
sample manifest
dataset loader
metric functions
robustness transforms
export helpers
```

---

# 57. Default parameters

```python
USE_BYOD = False
BYOD_PATH = ""
BYOD_LABELS_PATH = ""

SAMPLE_SEED = 42
SAMPLE_PER_CLASS = 20

RUN_ROBUSTNESS = True
RUN_SELECTIVE_CLASSIFICATION = False

OUTPUT_DIR = "outputs/document_type_classification"
```

No default path requires interaction.

---

# 58. Notebook cell plan

| # | Type | Section | Default |
|---:|---|---|---|
| 0 | Markdown | Title and objectives | Yes |
| 1 | Markdown | Document classification vs QA/OCR | Yes |
| 2 | Markdown | DiT architecture and document-domain pretraining | Yes |
| 3 | Code | Form parameters | Yes |
| 4 | Markdown | Runtime | Yes |
| 5 | Code | Install/check pins | Yes |
| 6 | Markdown | Model provenance + conversion boundary | Yes |
| 7 | Code | Converted SafeTensors manifest | Yes |
| 8 | Markdown | RVL-CDIP provenance | Yes |
| 9 | Code | Fetch/verify balanced 320-page test sample | Yes |
| 10 | Markdown | Class ordering | Yes |
| 11 | Code | Dataset validation + class counts | Yes |
| 12 | Markdown | Baselines | Yes |
| 13 | Code | Random/majority baseline | Yes |
| 14 | Markdown | Model preprocessing | Yes |
| 15 | Code | Load verified DiT | Yes |
| 16 | Markdown | What to inspect before prediction | Yes |
| 17 | Code | Warm-up + 320-page inference | Yes |
| 18 | Markdown | Metrics | Yes |
| 19 | Code | Accuracy/macro-F1/top-3 | Yes |
| 20 | Markdown | Per-class behavior | Yes |
| 21 | Code | Per-class metrics | Yes |
| 22 | Markdown | Confusion matrix exercise | Yes |
| 23 | Code | Confusion matrix + strongest pairs | Yes |
| 24 | Markdown | Error analysis | Yes |
| 25 | Code | High-confidence/low-margin examples | Yes |
| 26 | Markdown | Robustness concepts | Yes |
| 27 | Code | Select deterministic 16-page probe | Yes |
| 28 | Code | Rotation experiment | Yes |
| 29 | Code | Resolution experiment | Yes |
| 30 | Code | Crop experiment | Yes |
| 31 | Markdown | Robustness interpretation | Yes |
| 32 | Code | Robustness summary | Yes |
| 33 | Markdown | Confidence and score semantics | Yes |
| 34 | Code | Optional selective-classification experiment | No-op |
| 35 | Markdown | Visual galleries | Yes |
| 36 | Code | Correct/error galleries | Yes |
| 37 | Markdown | Machine-readable exports | Yes |
| 38 | Code | CSV + JSON + provenance | Yes |
| 39 | Markdown | BYOD | Yes |
| 40 | Code | Optional BYOD | No-op |
| 41 | Markdown | Limitations and transfer | Yes |
| 42 | Code | Terminal summary + output assertions | Yes |

Every executable cell must be preceded by explanatory markdown.

---

# 59. Required assertions

The canonical path should fail if:

```text
model identity does not match
model revision does not match

converted weight digest does not match
config digest does not match
processor digest does not match

16 classes are not present
class ordering differs from config

sample count != 320
class count != 20 each
sample image digest mismatches
duplicate sample image detected

model output shape != [batch,16]
scores contain non-finite values
score vector does not sum approximately to 1
predicted class outside 0..15

class_scores export does not contain exactly 16 rows per page

required output files are missing
```

The notebook MUST NOT assert:

```text
accuracy >= 92.11%
specific confusion pairs must appear
rotation must reduce accuracy
crop must reduce accuracy
a particular class must be easiest/hardest
```

Those are empirical observations.

---

# 60. Release verification

Preferred clean qualification environment:

```text
Kaggle CPU or Tesla T4
Python 3.12
```

Qualification must use:

- exact committed notebook;
- empty model cache;
- empty sample cache;
- no repository checkout;
- no credentials;
- no DIMER service.

Record:

```text
notebook commit
notebook blob SHA

Python
PyTorch
Transformers
device

model ID
model revision

source .bin digest
converted safetensors digest
conversion parity evidence

sample dataset revision
sample digest
320 image digests
20 images per class

majority baseline
accuracy
macro F1
top-3 accuracy
per-class results
confusion matrix

robustness observations
timings
cell success count
wall time
output inventory
```

Static validation is not runtime evidence.

---

# 61. Model repository recommendation

Create a dedicated carrier:

```text
kurtvalcorza/dit-document-classification-pipeline
```

Suggested DIMER listing:

```text
DiT Base Document-Type Classification — RVL-CDIP
```

Task:

```text
Image Classification - Document Type
```

---

# 62. Workshop repository placement

Because this notebook is task-focused and model-specific, either of these structures works.

Preferred if the dedicated carrier is built:

```text
dit-document-classification-pipeline/
  tutorials/
    dit_document_classification_rvlcdip.ipynb
```

If workshops remain centrally managed:

```text
ml-worker/
  integrations/
    dimer/
      workshops/
        document-type-classification/
          DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb
          README.md
          docs/
            release-verification.md
          tools/
            build_notebook.py
            validate_notebook.py
```

I would favor the **dedicated carrier repository** because this task should ultimately become a real DIMER model profile, not merely a cross-model workshop.

---

# 63. Relationship to the Document Intelligence track

The resulting track becomes:

```text
Document Intelligence
│
├── Document-Type Classification
│     └── DiT + RVL-CDIP          ← this notebook
│
├── Document Question Answering
│     ├── LayoutLM
│     └── Pix2Struct DocVQA
│
├── Document Extraction / OCR
│     ├── SmolDocling
│     ├── GOT-OCR
│     └── Florence-style extraction
│
└── Table Intelligence
      ├── Table Transformer Detection
      ├── Table Structure Recognition
      ├── TAPAS
      └── DePlot
```

---

# 64. Workshop learning arc

The intended teaching sequence is:

**A page is treated as an image**  
↓  
*No OCR is required for the classifier.*

**Document-domain pretraining**  
↓  
*DiT learns document-page visual structure from large-scale document imagery.*

**Fixed 16-class task**  
↓  
*The model predicts one RVL-CDIP class.*

**Full score vector**  
↓  
*Top-1 alone hides plausible competing classes.*

**Confusion analysis**  
↓  
*Visually related document types can fail in systematic ways.*

**Robustness analysis**  
↓  
*Orientation, cropping and source resolution alter the visual evidence.*

**Deployment boundary**  
↓  
*The model recognizes only its 16 trained document types.*

**Transfer lesson**  
↓  
*New organizational document categories require supervised adaptation rather than relabeling the existing output head.*