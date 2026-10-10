#!/usr/bin/env bash
# full chain for one config: prepare -> AE -> latent cache -> SPADE -> SPADE cache -> diffusion -> evaluation
# usage: bash scripts/run_all.sh configs/base.yaml [run_name]
set -euo pipefail
C=${1:?config}; NAME=${2:-diffusion}
OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open('$C'))['out_dir'])")
python scripts/inspect_volumes.py   --config "$C"
python scripts/prepare_data.py      --config "$C"
python scripts/train_ae3d.py        --config "$C" --resume
python scripts/cache_latents3d.py   --config "$C" --stage ae
python scripts/train_spade3d.py     --config "$C" --resume
python scripts/cache_latents3d.py   --config "$C" --stage spade
python scripts/train_diffusion3d.py --config "$C" --name "$NAME" --resume
python scripts/infer_evaluate3d.py  --config "$C" --ckpt "$OUT/$NAME/best.pth" --ceiling
