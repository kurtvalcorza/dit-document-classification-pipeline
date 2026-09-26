# Weights and conversion provenance

## Pinned upstream source

- Model: `microsoft/dit-base-finetuned-rvlcdip`
- Revision: `23f8b03d130fb66bbc9a15df3c75d753e49240eb`
- Source: `pytorch_model.bin`
- Bytes: `343362393`
- SHA-256: `1b7a901642c36ec7e32de997683223faf02028624fb2e62a7e7e798a5dd1344e`
- Serialization: PyTorch pickle/state dict
- Serving: **never**

The Hugging Face file inspection reports only the standard tensor reconstruction imports (`collections.OrderedDict`, `torch.FloatStorage`, `torch._utils._rebuild_tensor_v2`). The conversion tool still requires the exact source digest before loading, statically audits the archive's `data.pkl`, and uses `torch.load(..., weights_only=True)`.

## Derived serving artifact

Run:

```bash
python tools/convert_weights.py --download --remove-source
```

The tool downloads `config.json`, `preprocessor_config.json`, and the pinned source; verifies the source size and SHA-256; converts every state-dict tensor to contiguous CPU storage; writes `model.safetensors`; reloads it; checks tensor-name/shape/dtype/value identity; builds source and converted models from the local config; runs a fixed 224×224 probe through the pinned processor; and requires identical class IDs plus max absolute logit difference ≤ `1e-6`.

It then writes `dimer-base-manifest.json` with the source and derived digests, conversion tool versions, static-audit evidence, parity evidence, and the three files that constitute the serving snapshot. With `--remove-source`, the final snapshot contains no pickle.

## License gate

Conversion is a technical packaging step, not a licensing decision. DIMER hosting stays on **HOLD** until model-weight redistribution rights are explicitly documented.
