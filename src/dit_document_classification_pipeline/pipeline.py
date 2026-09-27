"""DIMER carrier for Microsoft DiT document-type classification on RVL-CDIP.

The runtime deliberately loads only a digest-verified SafeTensors snapshot. The upstream
``pytorch_model.bin`` is provenance input to ``tools/convert_weights.py`` and is never a
serving artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "microsoft/dit-base-finetuned-rvlcdip"
MODEL_REVISION = "23f8b03d130fb66bbc9a15df3c75d753e49240eb"
MODEL_KEY = "dit-base-finetuned-rvlcdip"
MODEL_LICENSE = "review-required"
SOURCE_FILENAME = "pytorch_model.bin"
SOURCE_BYTES = 343_362_393
SOURCE_SHA256 = "1b7a901642c36ec7e32de997683223faf02028624fb2e62a7e7e798a5dd1344e"
SERVING_FILENAME = "model.safetensors"
MANIFEST_NAME = "dimer-base-manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY

CLASS_NAMES = (
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
)
NUM_CLASSES = len(CLASS_NAMES)
MIN_IMAGE_SIDE = 32
MAX_IMAGE_SIDE = 4096
MAX_PIXELS = 32_000_000
MAX_BATCH = 64
DEFAULT_TOP_K = 3
DECISION_RULE = "argmax(logits)"

INPUT_SCHEMA: dict[str, Any] = {
    "input": "one PIL document-page image or a sequence of them; converted to RGB",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "max_pixels": MAX_PIXELS,
    "batch": [1, MAX_BATCH],
    "classes": list(CLASS_NAMES),
    "decision_rule": DECISION_RULE,
    "preprocessing": "pinned AutoImageProcessor from preprocessor_config.json (224 px, mean/std 0.5)",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Verify the converted serving snapshot and its derivation record."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")

    source = manifest.get("source") or {}
    if source.get("path") != SOURCE_FILENAME:
        raise ValueError("manifest source path does not name pytorch_model.bin")
    if source.get("bytes") != SOURCE_BYTES or source.get("sha256") != SOURCE_SHA256:
        raise ValueError("manifest source identity does not match the pinned upstream checkpoint")
    if source.get("served") is not False:
        raise ValueError("manifest must state source.served=false")

    derived = manifest.get("derived") or {}
    if derived.get("path") != SERVING_FILENAME:
        raise ValueError("manifest derived path does not name model.safetensors")
    if derived.get("derivedFromSha256") != SOURCE_SHA256:
        raise ValueError("derived asset is not bound to the pinned source digest")
    if not isinstance(derived.get("bytes"), int) or derived["bytes"] <= 0:
        raise ValueError("derived SafeTensors byte size is not recorded")
    if not isinstance(derived.get("sha256"), str) or len(derived["sha256"]) != 64:
        raise ValueError("derived SafeTensors SHA-256 is not recorded")

    required = {"config.json", "preprocessor_config.json", SERVING_FILENAME}
    records = {item["path"]: item for item in manifest.get("files", [])}
    if required - records.keys():
        raise ValueError(f"manifest missing required files: {sorted(required - records.keys())}")

    for name in sorted(required):
        record = records[name]
        file_path = root / name
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != record.get("bytes"):
            raise ValueError(f"{name}: size {size} != manifest {record.get('bytes')}")
        digest = sha256_file(file_path)
        if digest != record.get("sha256"):
            raise ValueError(f"{name}: sha256 {digest} != manifest {record.get('sha256')}")

    serving = root / SERVING_FILENAME
    if serving.stat().st_size != derived["bytes"] or sha256_file(serving) != derived["sha256"]:
        raise ValueError("derived asset record does not match model.safetensors")

    parity = manifest.get("conversion", {}).get("parity", {})
    if parity.get("predictedClassIdsIdentical") is not True:
        raise ValueError("conversion parity does not record identical predicted class IDs")
    difference = parity.get("maxAbsLogitDiff")
    tolerance = parity.get("tolerance")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in (difference, tolerance)
    ):
        raise ValueError("conversion parity must record numeric difference and tolerance")
    # The fixed ceiling also rejects infinity; chained bounds reject NaN and negatives.
    if not 0 <= difference <= tolerance <= 1e-6:
        raise ValueError("conversion parity must satisfy 0 <= difference <= tolerance <= 1e-6")

    return {
        "path": str(root),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "serving_file": SERVING_FILENAME,
        "serving_sha256": derived["sha256"],
        "source_sha256": SOURCE_SHA256,
        "license_review": manifest.get("licenseReview", "required"),
    }


def _check_images(images: Any) -> list[Image.Image]:
    if isinstance(images, Image.Image):
        images = [images]
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if not 1 <= len(images) <= MAX_BATCH:
        raise ValueError(f"batch size must be 1..{MAX_BATCH}, got {len(images)}")
    checked: list[Image.Image] = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"each image must be a PIL.Image.Image, got {type(image).__name__}")
        width, height = image.size
        if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
            raise ValueError(
                f"image size {image.size} outside {MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE} px per side"
            )
        if width * height > MAX_PIXELS:
            raise ValueError(f"image has {width * height:,} pixels; maximum is {MAX_PIXELS:,}")
        checked.append(image.convert("RGB"))
    return checked


def validate_inputs(
    images: Image.Image | Sequence[Image.Image], *, names: Sequence[str] | None = None
) -> dict[str, Any]:
    checked = _check_images(images)
    originals = [images] if isinstance(images, Image.Image) else list(images)
    if names is not None and len(names) != len(checked):
        raise ValueError("names must contain one entry per image")
    return {
        "schema": INPUT_SCHEMA,
        "inputs": [
            {
                "id": names[i] if names else f"image-{i}",
                "mode": original.mode,
                "width": original.size[0],
                "height": original.size[1],
                "pixels": original.size[0] * original.size[1],
            }
            for i, original in enumerate(originals)
        ],
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def validate_dataset(records: Sequence[Mapping[str, Any]], *, require_labels: bool = True) -> dict[str, Any]:
    if not 1 <= len(records) <= 5000:
        raise ValueError("dataset must contain 1..5000 records")
    seen: set[str] = set()
    counts = {label: 0 for label in CLASS_NAMES}
    for record in records:
        rid = record.get("id")
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise ValueError(f"record id must be a unique non-empty string, got {rid!r}")
        seen.add(rid)
        image = record.get("image")
        if not isinstance(image, Image.Image):
            raise TypeError("each dataset record must contain a single PIL.Image.Image")
        _check_images(image)
        label = record.get("label")
        if require_labels:
            if label not in counts:
                raise ValueError(f"label {label!r} is not an RVL-CDIP class")
            counts[label] += 1
    return {
        "records": len(records),
        "class_counts": {k: v for k, v in counts.items() if v},
        "labels_present": sum(v > 0 for v in counts.values()),
        "verdict": "accepted",
    }


def classification_metrics(
    truth: Sequence[int], predicted: Sequence[int], top3: Sequence[Sequence[int]] | None = None
) -> dict[str, Any]:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("truth and predicted must be non-empty and equal length")
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(truth, predicted, strict=True):
        if t not in range(NUM_CLASSES) or p not in range(NUM_CLASSES):
            raise ValueError("class IDs must be in 0..15")
        matrix[t, p] += 1
    per_class: list[dict[str, Any]] = []
    f1s: list[float] = []
    for i, label in enumerate(CLASS_NAMES):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        support = int(matrix[i, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class.append(
            {
                "class_id": i,
                "class_label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    result: dict[str, Any] = {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "n": len(truth),
    }
    if top3 is not None:
        if len(top3) != len(truth):
            raise ValueError("top3 must contain one ranked class list per item")
        result["top3_accuracy"] = float(
            np.mean([t in tuple(ranked)[:3] for t, ranked in zip(truth, top3, strict=True)])
        )
    return result


def _canonical_labels(config: Any) -> tuple[str, ...]:
    labels = tuple(config.id2label[i] for i in range(config.num_labels))
    if labels != CLASS_NAMES:
        raise ValueError(f"config label order differs from canonical RVL-CDIP order: {labels}")
    return labels


class DiTDocumentClassificationPipeline:
    """Local-only DiT inference over the fixed 16-class RVL-CDIP vocabulary."""

    def __init__(self, model: Any, processor: Any, *, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        _canonical_labels(model.config)

    @classmethod
    def from_pretrained(
        cls, *, weights_dir: str | Path | None = None, device: str | None = None
    ) -> DiTDocumentClassificationPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        verify_snapshot(root)
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

        config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        _canonical_labels(config)
        processor = AutoImageProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        model = AutoModelForImageClassification.from_config(config, trust_remote_code=False)
        state = load_file(str(root / SERVING_FILENAME), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"SafeTensors state-dict mismatch; missing={missing}, unexpected={unexpected}")
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(resolved).eval()
        return cls(model, processor, device=resolved)

    def predict(
        self, images: Image.Image | Sequence[Image.Image], *, top_k: int = DEFAULT_TOP_K
    ) -> dict[str, Any]:
        checked = _check_images(images)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= NUM_CLASSES:
            raise ValueError(f"top_k must be an integer in 1..{NUM_CLASSES}")
        import torch

        encoded = self.processor(images=checked, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
            scores = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = []
        for row in scores:
            order = np.argsort(-row)
            best = int(order[0])
            predictions.append(
                {
                    "predicted_class_id": best,
                    "predicted_label": CLASS_NAMES[best],
                    "top1_score": float(row[best]),
                    "top_k": [
                        {"class_id": int(i), "label": CLASS_NAMES[int(i)], "score": float(row[i])}
                        for i in order[:top_k]
                    ],
                    "class_scores": [float(v) for v in row],
                }
            )
        return {
            "predictions": predictions,
            "top_k": top_k,
            "decision_rule": DECISION_RULE,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": self.device,
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        validate_dataset(records, require_labels=True)
        truth: list[int] = []
        predicted: list[int] = []
        top3: list[list[int]] = []
        for start in range(0, len(records), MAX_BATCH):
            batch = records[start : start + MAX_BATCH]
            result = self.predict([r["image"] for r in batch], top_k=3)
            for record, pred in zip(batch, result["predictions"], strict=True):
                truth.append(CLASS_NAMES.index(record["label"]))
                predicted.append(pred["predicted_class_id"])
                top3.append([item["class_id"] for item in pred["top_k"]])
        metrics = classification_metrics(truth, predicted, top3)
        return {
            **metrics,
            "sample_kind": "labelled document-page classification set",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "verdict": "sample-sanity",
        }
