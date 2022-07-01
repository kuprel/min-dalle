#!/bin/bash

set -e

pip install torch

mkdir -p ./pretrained/dalle_bart_mega/
curl https://huggingface.co/kuprel/min-dalle/resolve/main/vocab.json -L --output ./pretrained/dalle_bart_mega/vocab.json
curl https://huggingface.co/kuprel/min-dalle/resolve/main/merges.txt -L --output ./pretrained/dalle_bart_mega/merges.txt
curl https://huggingface.co/kuprel/min-dalle/resolve/main/config.json -L --output ./pretrained/dalle_bart_mega/config.json
curl https://huggingface.co/kuprel/min-dalle/resolve/main/encoder.pt -L --output ./pretrained/dalle_bart_mega/encoder.pt
curl https://huggingface.co/kuprel/min-dalle/resolve/main/decoder.pt -L --output ./pretrained/dalle_bart_mega/decoder.pt

mkdir -p ./pretrained/dalle_bart_mini/
curl https://huggingface.co/kuprel/min-dalle/resolve/main/vocab_mini.json -L --output ./pretrained/dalle_bart_mini/vocab.json
curl https://huggingface.co/kuprel/min-dalle/resolve/main/merges_mini.txt -L --output ./pretrained/dalle_bart_mini/merges.txt
curl https://huggingface.co/kuprel/min-dalle/resolve/main/config_mini.json -L --output ./pretrained/dalle_bart_mini/config.json
curl https://huggingface.co/kuprel/min-dalle/resolve/main/encoder_mini.pt -L --output ./pretrained/dalle_bart_mini/encoder.pt
curl https://huggingface.co/kuprel/min-dalle/resolve/main/decoder_mini.pt -L --output ./pretrained/dalle_bart_mini/decoder.pt

mkdir -p ./pretrained/vqgan/
curl https://huggingface.co/kuprel/min-dalle/resolve/main/detoker.pt -L --output ./pretrained/vqgan/detoker.pt