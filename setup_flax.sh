#!/bin/bash

set -e

pip install torch flax wandb

# download vqgan
mkdir -p pretrained/vqgan
curl https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/resolve/main/flax_model.msgpack -L --output ./pretrained/vqgan/flax_model.msgpack

# download dalle-mini and dalle-mega
python3 -m wandb login --anonymously
python3 -m wandb artifact get --root=./pretrained/dalle_bart_mini dalle-mini/dalle-mini/mini-1:v0
python3 -m wandb artifact get --root=./pretrained/dalle_bart_mega dalle-mini/dalle-mini/mega-1-fp16:v14 
