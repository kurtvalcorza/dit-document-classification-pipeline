"""Model stages of the DiT document-type classification workshop notebook.

The notebook carries this file byte-for-byte (cell ``uvcarrier``) and runs every stage that needs
PyTorch, Transformers, datasets or the Hugging Face Hub as a subprocess on the Python of an isolated
uv environment::

    python document_classification_workshop.py --config stage_config.json --stage <name> [options]

Nothing here is imported into the notebook kernel. The kernel keeps the metrics, tables, galleries and
exports, which need only NumPy and Pillow; stages hand results back as JSON, ``.npy`` arrays and
lossless PNG pages under ``<work_dir>/state`` and ``<work_dir>/pages``.

The function bodies are the notebook's former in-kernel cells (442373b0, 3eaee889, d954e7b8,
ca003003, 18aafd5c, fe55f221, b485b2eb, 3891d68d) with the display code left in the notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

STAGES = ("environment", "download", "convert", "dataset", "sample", "load-check", "predict")

PINS = {
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "torchaudio": "2.11.0",
    "transformers": "4.57.6",
    "safetensors": "0.8.0",
    "numpy": "2.1.3",
    "pillow": "11.3.0",
    "huggingface-hub": "0.36.2",
    "datasets": "4.1.1",
}

MODEL_ID = "microsoft/dit-base-finetuned-rvlcdip"
MODEL_REVISION = "23f8b03d130fb66bbc9a15df3c75d753e49240eb"
SOURCE_BIN_SHA256 = "1b7a901642c36ec7e32de997683223faf02028624fb2e62a7e7e798a5dd1344e"

SOURCE_DIR = Path("weights/dit-rvlcdip-source")
CONVERTED_DIR = Path("weights/dit-rvlcdip-safetensors")

LABELS = [
    "letter","form","email","handwritten","advertisement","scientific report",
    "scientific publication","specification","file folder","news article","budget",
    "invoice","presentation","questionnaire","resume","memo"
]

ROBUSTNESS_VARIANTS = ["original","rotate90","rotate180","low_resolution","center_crop"]

# Set by main() from the notebook's stage configuration.
CONFIG: dict = {}
WORK = Path("work/document_type_classification")


# ---- shared helpers -------------------------------------------------------------------------------------
def installed_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def matches_public_version(observed, expected):
    from packaging.version import InvalidVersion, Version

    if observed is None:
        return False
    try:
        return Version(Version(str(observed)).public) == Version(Version(str(expected)).public)
    except InvalidVersion:
        return False


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_state(name, value):
    path = WORK / "state" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def read_state(name):
    return json.loads((WORK / "state" / name).read_text(encoding="utf-8"))


def device():
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---- environment ----------------------------------------------------------------------------------------
def verify_environment():
    """Every pinned distribution in this environment has its pinned public version."""
    observed = {k: installed_version(k) for k in PINS}
    bad = {k: (observed[k], v) for k, v in PINS.items() if not matches_public_version(observed[k], v)}
    if bad:
        raise RuntimeError(f"Isolated environment does not match the pins: {bad}")
    return observed


def stage_environment():
    verify_environment()
    import datasets
    import huggingface_hub
    import numpy as np
    import torch
    import torchvision
    import transformers

    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "numpy": np.__version__,
        "pillow": importlib_metadata.version("pillow"),
        "device": device(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        runtime["gpu_name"] = torch.cuda.get_device_name(0)
        runtime["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    write_state("runtime.json", runtime)
    print(runtime)


# ---- model identity, download and conversion -------------------------------------------------------------
def download_and_verify_checkpoint():
    """Download the pinned files and refuse a source checkpoint whose SHA-256 differs from the pin."""
    from huggingface_hub import hf_hub_download

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
            revision=MODEL_REVISION,
            local_dir=str(SOURCE_DIR),
        )

    source_bin = SOURCE_DIR / "pytorch_model.bin"
    actual_source_sha = sha256_file(source_bin)
    if actual_source_sha != SOURCE_BIN_SHA256:
        raise RuntimeError(
            f"Source checkpoint SHA-256 {actual_source_sha} != pinned {SOURCE_BIN_SHA256}; refusing to load"
        )
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "source_bin_bytes": source_bin.stat().st_size,
        "source_bin_sha256": actual_source_sha,
        "config_sha256": sha256_file(SOURCE_DIR / "config.json"),
        "processor_sha256": sha256_file(SOURCE_DIR / "preprocessor_config.json"),
    }


def stage_download():
    record = download_and_verify_checkpoint()
    write_state("download.json", record)
    print(record)


def make_probe(kind):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(image)
    if kind == "letter":
        for y in range(140, 980, 45):
            draw.line((120,y,760,y), fill="black", width=3)
        draw.rectangle((110,80,420,115), outline="black", width=3)
    else:
        for y in range(100, 1050, 80):
            draw.rectangle((100,y,800,y+35), outline="black", width=2)
        draw.line((450,100,450,1085), fill="black", width=2)
    return image


def convert_and_verify():
    """Convert the verified pickle once to SafeTensors and prove tensor identity and logit parity."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    downloaded = read_state("download.json")
    source_bin = SOURCE_DIR / "pytorch_model.bin"
    if sha256_file(source_bin) != SOURCE_BIN_SHA256:
        raise RuntimeError("Source checkpoint changed after verification; run the download cell again")

    # Copy load-bearing non-weight assets into the converted snapshot.
    shutil.copy2(SOURCE_DIR / "config.json", CONVERTED_DIR / "config.json")
    shutil.copy2(SOURCE_DIR / "preprocessor_config.json", CONVERTED_DIR / "preprocessor_config.json")

    config = AutoConfig.from_pretrained(str(SOURCE_DIR), local_files_only=True)
    actual_labels = [config.id2label[i] for i in range(16)]
    if actual_labels != LABELS:
        raise RuntimeError(f"Config class ordering changed: {actual_labels}")

    state = torch.load(source_bin, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Source checkpoint did not decode to a non-empty state dict")

    source_model = AutoModelForImageClassification.from_config(config)
    missing, unexpected = source_model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Strict state identity failed; missing={missing}, unexpected={unexpected}")
    source_model.eval()

    converted_path = CONVERTED_DIR / "model.safetensors"
    cpu_state = {k: v.detach().cpu().contiguous() for k,v in source_model.state_dict().items()}
    save_file(cpu_state, str(converted_path))

    converted_sha256 = sha256_file(converted_path)
    conversion_manifest = {
        "format": "dimer_derived_model_snapshot",
        "formatVersion": 1,
        "modelKey": "dit-base-finetuned-rvlcdip",
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "source": {
            "file": "pytorch_model.bin",
            "bytes": source_bin.stat().st_size,
            "sha256": SOURCE_BIN_SHA256,
            "serialization": "PyTorch pickle; loaded only with weights_only=True",
        },
        "derived": {
            "file": "model.safetensors",
            "bytes": converted_path.stat().st_size,
            "sha256": converted_sha256,
            "derivedFromSha256": SOURCE_BIN_SHA256,
        },
        "companions": {
            "config.json": downloaded["config_sha256"],
            "preprocessor_config.json": downloaded["processor_sha256"],
        },
        "conversion": {
            "torch": torch.__version__,
            "safetensors": importlib_metadata.version("safetensors"),
        },
        "redistributionStatus": "pending-review",
        "hostingStatus": "HOLD",
    }
    (CONVERTED_DIR / "dimer-base-manifest.json").write_text(
        json.dumps(conversion_manifest, indent=2), encoding="utf-8"
    )

    processor_probe = AutoImageProcessor.from_pretrained(str(CONVERTED_DIR), local_files_only=True)
    converted_model_probe = AutoModelForImageClassification.from_pretrained(
        str(CONVERTED_DIR), local_files_only=True
    ).eval()

    probe_images = [make_probe("letter"), make_probe("form")]
    probe_inputs = processor_probe(images=probe_images, return_tensors="pt")

    with torch.inference_mode():
        source_logits = source_model(**probe_inputs).logits
        converted_logits = converted_model_probe(**probe_inputs).logits

    max_abs_logit_diff = float((source_logits - converted_logits).abs().max())
    if max_abs_logit_diff > 1e-6:
        raise RuntimeError(f"Converted logit parity failed: max abs diff={max_abs_logit_diff}")

    if list(source_model.state_dict()) != list(converted_model_probe.state_dict()):
        raise RuntimeError("Converted state-dict key ordering differs from source model")

    for key in source_model.state_dict():
        a, b = source_model.state_dict()[key], converted_model_probe.state_dict()[key]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            raise RuntimeError(f"Converted tensor mismatch: {key}")

    return {
        "converted_bytes": converted_path.stat().st_size,
        "converted_sha256": converted_sha256,
        "parity_max_abs_logit_diff": max_abs_logit_diff,
        "status": "candidate conversion verified",
    }


def stage_convert():
    record = convert_and_verify()
    write_state("convert.json", record)
    print(record)


# ---- dataset and the deterministic tutorial sample -------------------------------------------------------
def load_pinned_test_split():
    from datasets import load_dataset

    dataset_test = load_dataset(
        CONFIG["dataset_id"],
        split="test",
        revision=CONFIG["dataset_revision"],
    )
    if len(dataset_test) != 992:
        raise RuntimeError(f"Expected 992 rows in the pinned derived test split, got {len(dataset_test)}")
    return dataset_test


def stage_dataset():
    dataset_test = load_pinned_test_split()
    record = {
        "dataset_id": CONFIG["dataset_id"],
        "dataset_revision": CONFIG["dataset_revision"],
        "test_rows": len(dataset_test),
    }
    write_state("dataset.json", record)
    print(record)


def select_sample_indices(labels, sample_seed, sample_per_class):
    """Rank rows within each class by SHA-256 of seed:row:label and keep the first rows of each class."""
    ranked = defaultdict(list)
    for row_index, label_id in enumerate(labels):
        label_id = int(label_id)
        key = hashlib.sha256(f"{sample_seed}:{row_index}:{label_id}".encode()).hexdigest()
        ranked[label_id].append((key,row_index))

    selected_indices = []
    for label_id in range(16):
        items = sorted(ranked[label_id])
        if len(items) < sample_per_class:
            raise RuntimeError(f"Class {label_id} has only {len(items)} rows")
        selected_indices.extend(idx for _key,idx in items[:sample_per_class])
    return sorted(selected_indices)


def image_bytes_and_rgb(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        image = value.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        return data, image
    if isinstance(value, dict) and value.get("bytes") is not None:
        data = value["bytes"]
        image = Image.open(io.BytesIO(data))
        image.load()
        return data, image.convert("RGB")
    raise TypeError(f"Unsupported dataset image type: {type(value).__name__}")


def pixel_sha256(image):
    return hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def build_sample(dataset_test, page_dir):
    """Decode, validate and digest the selected pages; write each as a lossless RGB PNG for the kernel."""
    sample_seed, sample_per_class = CONFIG["sample_seed"], CONFIG["sample_per_class"]
    min_side, max_side, max_pixels = CONFIG["min_image_side"], CONFIG["max_image_side"], CONFIG["max_pixels"]
    labels = [int(label) for label in dataset_test["label"]]
    selected_indices = select_sample_indices(labels, sample_seed, sample_per_class)

    page_dir = Path(page_dir)
    if page_dir.exists():
        shutil.rmtree(page_dir)
    page_dir.mkdir(parents=True)
    sample_records = []
    seen_pixel_digests = set()

    for row_index in selected_indices:
        row = dataset_test[row_index]
        raw_bytes, image = image_bytes_and_rgb(row["image"])
        label_id = int(row["label"])
        if label_id not in range(16):
            raise RuntimeError(f"Invalid class id {label_id}")
        if not (min_side <= min(image.size) and max(image.size) <= max_side):
            raise RuntimeError(f"Row {row_index}: dimensions {image.size} outside tutorial limits")
        if image.width * image.height > max_pixels:
            raise RuntimeError(f"Row {row_index}: image exceeds {max_pixels} pixels")
        pixel_digest = pixel_sha256(image)
        if pixel_digest in seen_pixel_digests:
            raise RuntimeError(f"Duplicate decoded page detected at source row {row_index}")
        seen_pixel_digests.add(pixel_digest)
        image_id = f"rvl-derived-test-{row_index:04d}"
        png = page_dir / f"{image_id}.png"
        image.save(png, format="PNG")
        sample_records.append({
            "image_id": image_id,
            "source_row_index": row_index,
            "png": str(png),
            "source_bytes_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "pixel_sha256": pixel_digest,
            "width": image.width,
            "height": image.height,
            "label_id": label_id,
            "label": LABELS[label_id],
        })

    counts = Counter(r["label_id"] for r in sample_records)
    if (len(sample_records) != 16 * sample_per_class or set(counts.values()) != {sample_per_class}
            or len(counts) != 16):
        raise RuntimeError(f"Balanced sample validation failed: {counts}")

    sample_digest_payload = "\n".join(
        json.dumps(
            [r["source_row_index"], r["label_id"], r["pixel_sha256"]],
            separators=(",",":"),
        )
        for r in sorted(sample_records, key=lambda x:x["source_row_index"])
    )
    sample_digest = hashlib.sha256(sample_digest_payload.encode()).hexdigest()
    return {"records": sample_records, "sample_digest": sample_digest}


def stage_sample():
    sample = build_sample(load_pinned_test_split(), WORK / "pages" / "sample")
    write_state("sample.json", sample)
    print({"sample_images": len(sample["records"]), "sample_digest": sample["sample_digest"]})


# ---- model load and inference ----------------------------------------------------------------------------
def load_converted_model():
    """Load only the converted SafeTensors snapshot; the source pickle is never used for evaluation."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(str(CONVERTED_DIR), local_files_only=True)
    t0 = time.perf_counter()
    model = AutoModelForImageClassification.from_pretrained(
        str(CONVERTED_DIR),
        local_files_only=True,
    ).to(device()).eval()
    model_load_seconds = time.perf_counter() - t0
    config_labels = [model.config.id2label[i] for i in range(16)]
    if config_labels != LABELS:
        raise RuntimeError("Loaded class ordering differs from canonical RVL-CDIP order")
    return processor, model, model_load_seconds


def stage_load_check():
    _processor, model, model_load_seconds = load_converted_model()
    record = {
        "model_load_seconds": model_load_seconds,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "class_count": len(LABELS),
        "device": device(),
    }
    write_state("load_check.json", record)
    print(record)


def batched_predict(processor, model, records, batch_size):
    import numpy as np
    import torch

    DEVICE = device()
    all_logits = []
    timings = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warm-up.
    warm = processor(images=[records[0]["image"]], return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        _ = model(**warm).logits
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for start in range(0, len(records), batch_size):
        batch_records = records[start:start+batch_size]
        inputs = processor(
            images=[r["image"] for r in batch_records],
            return_tensors="pt",
        ).to(DEVICE)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
        if logits.shape != (len(batch_records), 16):
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced non-finite logits")
        all_logits.append(logits.float().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    scores = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Softmax score rows do not sum approximately to 1")
    peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    return logits, scores, timings, peak_gpu


def transform_variant(image, variant):
    from PIL import Image

    image = image.convert("RGB")
    if variant == "original":
        return image
    if variant == "rotate90":
        return image.rotate(-90, expand=True, fillcolor="white")
    if variant == "rotate180":
        return image.rotate(180, expand=True, fillcolor="white")
    if variant == "low_resolution":
        low = image.resize((112,112), Image.Resampling.BILINEAR)
        return low.resize(image.size, Image.Resampling.BILINEAR)
    if variant == "center_crop":
        w,h = image.size
        cw,ch = int(w*0.8), int(h*0.8)
        x0,y0 = (w-cw)//2, (h-ch)//2
        return image.crop((x0,y0,x0+cw,y0+ch))
    raise ValueError(variant)


def stage_predict(job_dir):
    """Score the pages listed in <job_dir>/job.json; write logits.npy, scores.npy and result.json there.

    An item with a "variant" is transformed here first and the transformed page is written back as
    PNG, so the notebook displays exactly the page the model scored.
    """
    import numpy as np
    from PIL import Image

    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    records = []
    scored_paths = []
    for k, item in enumerate(job["items"]):
        with Image.open(item["path"]) as opened:
            image = opened.convert("RGB")
        variant = item.get("variant")
        if variant is not None:
            image = transform_variant(image, variant)
            path = job_dir / f"scored-{k:04d}.png"
            image.save(path, format="PNG")
            scored_paths.append(str(path))
        else:
            scored_paths.append(item["path"])
        records.append({"image": image})
    if not records:
        raise ValueError("No pages to score")
    processor, model, _seconds = load_converted_model()
    logits, scores, timings, peak_gpu = batched_predict(processor, model, records, int(job["batch_size"]))
    np.save(job_dir / "logits.npy", logits)
    np.save(job_dir / "scores.npy", scores)
    (job_dir / "result.json").write_text(json.dumps({
        "timings": timings,
        "peak_gpu_memory_bytes": peak_gpu,
        "scored_paths": scored_paths,
        "device": device(),
    }, indent=2), encoding="utf-8")
    print({"pages": len(records), "batches": len(timings), "device": device()})


# ---- command line ---------------------------------------------------------------------------------------
def main(argv=None):
    global CONFIG, WORK
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--job", help="job directory of the predict stage")
    args = parser.parse_args(argv)
    CONFIG = json.loads(Path(args.config).read_text(encoding="utf-8"))
    WORK = Path(CONFIG["work_dir"])
    if args.stage == "predict":
        if not args.job:
            parser.error("--job is required for --stage predict")
        stage_predict(args.job)
        return 0
    {
        "environment": stage_environment,
        "download": stage_download,
        "convert": stage_convert,
        "dataset": stage_dataset,
        "sample": stage_sample,
        "load-check": stage_load_check,
    }[args.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
