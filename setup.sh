#!/bin/bash

set -e

pip install -r requirements.txt

mkdir -p pretrained/vqgan

# download vqgan
curl https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/resolve/main/flax_model.msgpack -L --output ./pretrained/vqgan/flax_model.msgpack

# download dalle-mini and dalle mega
python -m wandb login --anonymously
python -m wandb artifact get --root=./pretrained/dalle_bart_mini dalle-mini/dalle-mini/mini-1:v0
python -m wandb artifact get --root=./pretrained/dalle_bart_mega dalle-mini/dalle-mini/mega-1-fp16:v14 
