#!/usr/bin/env python
"""Audit and convert the pinned Microsoft DiT PyTorch checkpoint to SafeTensors.

The source pickle is provenance only. The serving snapshot contains model.safetensors,
config.json, preprocessor_config.json, and dimer-base-manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dit_document_classification_pipeline import (  # noqa: E402
    CLASS_NAMES,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    SERVING_FILENAME,
    SOURCE_BYTES,
    SOURCE_FILENAME,
    SOURCE_SHA256,
    sha256_file,
)

TOLERANCE = 1e-6
ALLOWED_PICKLE_GLOBALS = {
    "collections.OrderedDict",
    "torch.FloatStorage",
    "torch._utils._rebuild_tensor_v2",
}


def audit_pickle_globals(path: Path) -> set[str]:
    """Statically inspect the PyTorch ZIP pickle without importing or executing it."""
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one data.pkl in checkpoint, found {candidates}")
        payload = archive.read(candidates[0])
    found: set[str] = set()
    for opcode, arg, _ in pickletools.genops(payload):
        if opcode.name == "GLOBAL" and isinstance(arg, str):
            module, _, name = arg.partition(" ")
            found.add(f"{module}.{name}")
    unexpected = found - ALLOWED_PICKLE_GLOBALS
    if unexpected:
        raise ValueError(f"unexpected pickle globals: {sorted(unexpected)}")
    return found


def _download(filename: str, root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
            revision=MODEL_REVISION,
            local_dir=str(root),
        )
    )


def _state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict) and obj and all(hasattr(v, "shape") for v in obj.values()):
        return dict(obj)
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        state = obj["state_dict"]
        if state and all(hasattr(v, "shape") for v in state.values()):
            return dict(state)
    raise ValueError("source checkpoint is not a tensor-only state dict")


def _record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def convert(root: Path, *, download: bool = False, remove_source: bool = False) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "preprocessor_config.json", SOURCE_FILENAME):
        if not (root / filename).is_file():
            if not download:
                raise FileNotFoundError(f"missing {filename}; rerun with --download")
            _download(filename, root)

    source = root / SOURCE_FILENAME
    if source.stat().st_size != SOURCE_BYTES:
        raise ValueError(f"source size {source.stat().st_size} != pinned {SOURCE_BYTES}")
    digest = sha256_file(source)
    if digest != SOURCE_SHA256:
        raise ValueError(f"source sha256 {digest} != pinned {SOURCE_SHA256}")

    pickle_globals = audit_pickle_globals(source)

    import torch
    import transformers
    from safetensors import __version__ as safetensors_version
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    loaded = torch.load(source, map_location="cpu", weights_only=True)
    state = _state_dict(loaded)
    converted_state = {name: tensor.detach().cpu().clone().contiguous() for name, tensor in state.items()}
    output = root / SERVING_FILENAME
    save_file(converted_state, str(output))
    reloaded = load_file(str(output), device="cpu")
    if set(reloaded) != set(state):
        raise ValueError("SafeTensors conversion changed the tensor-name set")
    for name, tensor in state.items():
        other = reloaded[name]
        if tensor.shape != other.shape or tensor.dtype != other.dtype or not torch.equal(tensor.cpu(), other):
            raise ValueError(f"conversion mismatch for tensor {name}")

    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    labels = tuple(config.id2label[i] for i in range(config.num_labels))
    if labels != CLASS_NAMES:
        raise ValueError(f"config class ordering is not canonical: {labels}")
    processor = AutoImageProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    source_model = AutoModelForImageClassification.from_config(config, trust_remote_code=False)
    converted_model = AutoModelForImageClassification.from_config(config, trust_remote_code=False)
    source_model.load_state_dict(state, strict=True)
    converted_model.load_state_dict(reloaded, strict=True)
    source_model.eval()
    converted_model.eval()

    from PIL import Image

    probe = Image.new("RGB", (224, 224), (237, 237, 237))
    inputs = processor(images=probe, return_tensors="pt")
    with torch.inference_mode():
        source_logits = source_model(**inputs).logits
        converted_logits = converted_model(**inputs).logits
    max_diff = float((source_logits - converted_logits).abs().max().item())
    same_pred = bool(source_logits.argmax(-1).item() == converted_logits.argmax(-1).item())
    if not same_pred or max_diff > TOLERANCE:
        raise ValueError(f"conversion parity failed: same_pred={same_pred}, max_abs_logit_diff={max_diff}")

    files = [_record(root / "config.json"), _record(root / "preprocessor_config.json"), _record(output)]
    derived = _record(output)
    manifest = {
        "format": "dimer-derived-model-snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "licenseReview": "required",
        "redistributionStatus": "pending-review",
        "hostingStatus": "HOLD",
        "source": {
            "path": SOURCE_FILENAME,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "serialization": "PyTorch pickle",
            "served": False,
        },
        "derived": {
            **derived,
            "serialization": "SafeTensors",
            "derivedFromSha256": SOURCE_SHA256,
            "served": True,
        },
        "conversion": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors_version,
            "staticPickleGlobals": sorted(pickle_globals),
            "allowedPickleGlobals": sorted(ALLOWED_PICKLE_GLOBALS),
            "tensorNamesIdentical": True,
            "tensorShapesIdentical": True,
            "tensorDtypesIdentical": True,
            "parity": {
                "predictedClassIdsIdentical": same_pred,
                "maxAbsLogitDiff": max_diff,
                "tolerance": TOLERANCE,
            },
        },
        "files": files,
        "totalBytes": sum(item["bytes"] for item in files),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if remove_source:
        source.unlink()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "weights" / MODEL_KEY)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--remove-source", action="store_true")
    args = parser.parse_args()
    manifest = convert(args.snapshot_dir, download=args.download, remove_source=args.remove_source)
    print(json.dumps({"derived": manifest["derived"], "parity": manifest["conversion"]["parity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
