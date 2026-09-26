import hashlib
import json

import pytest
from PIL import Image

from dit_document_classification_pipeline import (
    CLASS_NAMES,
    MAX_BATCH,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_BYTES,
    SOURCE_FILENAME,
    SOURCE_SHA256,
    DiTDocumentClassificationPipeline,
    classification_metrics,
    validate_dataset,
    validate_inputs,
    verify_snapshot,
)


def test_validate_inputs_and_boundaries():
    image = Image.new("L", (224, 300))
    result = validate_inputs(image, names=["page.png"])
    assert result["verdict"] == "accepted"
    assert result["inputs"][0]["mode"] == "L"
    with pytest.raises(ValueError):
        validate_inputs(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 100)))
    with pytest.raises(ValueError):
        validate_inputs(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 100)))
    with pytest.raises(ValueError):
        validate_inputs([Image.new("RGB", (64, 64))] * (MAX_BATCH + 1))


def test_validate_dataset_requires_rvl_labels_and_unique_ids():
    records = [
        {"id": "a", "image": Image.new("RGB", (64, 64)), "label": "letter"},
        {"id": "b", "image": Image.new("RGB", (64, 64)), "label": "memo"},
    ]
    result = validate_dataset(records)
    assert result["records"] == 2 and result["labels_present"] == 2
    with pytest.raises(ValueError, match="RVL-CDIP"):
        validate_dataset([{**records[0], "label": "passport"}])
    with pytest.raises(ValueError, match="unique"):
        validate_dataset([records[0], records[0]])


def test_classification_metrics_top3_and_macro_f1():
    metrics = classification_metrics(
        [0, 1, 1, 15],
        [0, 0, 1, 15],
        [[0, 2, 1], [0, 1, 2], [1, 0, 2], [15, 1, 0]],
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["top3_accuracy"] == 1.0
    assert len(metrics["per_class"]) == len(CLASS_NAMES)
    assert metrics["confusion_matrix"][1][0] == 1


def _record(path):
    content = path.read_bytes()
    return {"path": path.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


@pytest.fixture
def converted_snapshot(tmp_path):
    config = tmp_path / "config.json"
    processor = tmp_path / "preprocessor_config.json"
    weights = tmp_path / "model.safetensors"
    config.write_text("{}", encoding="utf-8")
    processor.write_text("{}", encoding="utf-8")
    weights.write_bytes(b"safe")
    derived = _record(weights)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "licenseReview": "required",
        "source": {
            "path": SOURCE_FILENAME,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "served": False,
        },
        "derived": {**derived, "derivedFromSha256": SOURCE_SHA256},
        "conversion": {
            "parity": {
                "predictedClassIdsIdentical": True,
                "maxAbsLogitDiff": 0.0,
                "tolerance": 1e-6,
            }
        },
        "files": [_record(config), _record(processor), derived],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, manifest


def test_verify_snapshot_accepts_converted_only_serving_shape(converted_snapshot):
    tmp_path, manifest = converted_snapshot
    result = verify_snapshot(tmp_path)
    assert result["source_sha256"] == SOURCE_SHA256
    assert result["serving_sha256"] == manifest["derived"]["sha256"]
    manifest["source"]["served"] = True
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="served=false"):
        verify_snapshot(tmp_path)


@pytest.mark.parametrize(
    "parity",
    [
        {"maxAbsLogitDiff": 10, "tolerance": 100},
        {"maxAbsLogitDiff": 0, "tolerance": 100},
        {"maxAbsLogitDiff": float("nan"), "tolerance": 1e-6},
        {"maxAbsLogitDiff": float("inf"), "tolerance": 1e-6},
        {"maxAbsLogitDiff": -1, "tolerance": 1e-6},
        {"maxAbsLogitDiff": 0, "tolerance": float("nan")},
        {"maxAbsLogitDiff": 0, "tolerance": float("inf")},
        {"maxAbsLogitDiff": 0, "tolerance": -1},
        {"tolerance": float("inf")},
        {"maxAbsLogitDiff": 0},
        {"maxAbsLogitDiff": None, "tolerance": 1e-6},
        {"maxAbsLogitDiff": False, "tolerance": 1e-6},
        {"maxAbsLogitDiff": "0", "tolerance": 1e-6},
        {"maxAbsLogitDiff": 2e-6, "tolerance": 1e-6},
        {"maxAbsLogitDiff": 1e-6, "tolerance": 1e-7},
    ],
)
def test_verify_snapshot_rejects_invalid_parity(converted_snapshot, parity):
    root, manifest = converted_snapshot
    manifest["conversion"]["parity"] = {"predictedClassIdsIdentical": True, **parity}
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="parity"):
        verify_snapshot(root)


@pytest.mark.parametrize("difference,tolerance", [(0, 0), (1e-7, 1e-7), (1e-6, 1e-6)])
def test_verify_snapshot_accepts_parity_boundary(converted_snapshot, difference, tolerance):
    root, manifest = converted_snapshot
    manifest["conversion"]["parity"].update(maxAbsLogitDiff=difference, tolerance=tolerance)
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_snapshot(root)["model_id"] == MODEL_ID


@pytest.mark.parametrize("container", [list, tuple])
def test_dataset_and_evaluator_reject_image_batches(container):
    records = [{"id": "a", "image": container([Image.new("RGB", (64, 64))]), "label": "letter"}]
    with pytest.raises(TypeError, match="single PIL"):
        validate_dataset(records)
    # Validation must reject this before any model or processor is needed.
    pipeline = DiTDocumentClassificationPipeline.__new__(DiTDocumentClassificationPipeline)
    with pytest.raises(TypeError, match="single PIL"):
        pipeline.evaluate(records)
