#!/usr/bin/env bash

if ! [ -f /.dockerenv ]; then
    pip install -r requirements.txt
else
    echo "Running in container, skipping package installation..."
fi

# paths
DIR_MINI=./pretrained/dalle_bart_mini
DIR_MEGA=./pretrained/dalle_bart_mega
DIR_VQGAN=./pretrained/vqgan
DIR_GENERATED=./generated
VQGAN_FILE="$DIR_VQGAN/flax_model.msgpack"

mkdir -p "$DIR_VQGAN"
mkdir -p "$DIR_GENERATED"

# download vqgan
if ! [ -f "$VQGAN_FILE" ]; then
    curl https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/resolve/main/flax_model.msgpack -L --output "$VQGAN_FILE"
else
    echo "Skipping VQGAN download. Remove $VQGAN_FILE to redownload."
fi

wandb login --anonymously

# download dalle-mini (~1.7G on disk)
if ! [ -d $DIR_MINI ]; then
    wandb artifact get --root=./pretrained/dalle_bart_mini dalle-mini/dalle-mini/mini-1:v0
else
    echo "Skipping Dalle mini download. Remove $DIR_MINI to redownload."
fi

# download dalle mega (~4.9G on disk)
if ! [ -d "$DIR_MEGA" ]; then
    wandb artifact get --root=./pretrained/dalle_bart_mega dalle-mini/dalle-mini/mega-1-fp16:v14
else
    echo "Skipping Dalle mega download. Remove $DIR_MEGA to redownload."
fi
