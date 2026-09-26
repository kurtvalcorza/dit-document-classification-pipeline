# Local model snapshots

`weights/dit-base-finetuned-rvlcdip/` is generated locally by `tools/convert_weights.py` and should contain:

- `config.json`
- `preprocessor_config.json`
- `model.safetensors`
- `dimer-base-manifest.json`

The upstream `pytorch_model.bin` is provenance/conversion input only and must not remain in a DIMER serving snapshot.
