# Release status

**HOLD — implementation candidate; model-weight redistribution/licence review pending.**

The pipeline code is implemented for the pinned `microsoft/dit-base-finetuned-rvlcdip` revision and only serves a converted `model.safetensors` snapshot. The upstream `pytorch_model.bin` is provenance input to the conversion tool and is never a serving artifact.

Promotion requires all of the following:

1. run `tools/convert_weights.py --download --remove-source` in a trusted environment;
2. commit the generated manifest metadata (not the model weights) and record conversion parity;
3. run unit/static checks and a clean hosted notebook/runtime qualification;
4. document an explicit decision on model-weight redistribution/licensing before DIMER hosting.

No Release-grade or public-hosting claim is made by this branch.
