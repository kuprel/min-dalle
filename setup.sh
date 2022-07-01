#!/bin/bash

set -e

pip3 install -r requirements.txt

repo_path="https://huggingface.co/kuprel/min-dalle/resolve/main"

mini_path="./pretrained/dalle_bart_mini"
mega_path="./pretrained/dalle_bart_mega"
vqgan_path="./pretrained/vqgan"

mkdir -p ${vqgan_path}
mkdir -p ${mini_path}
mkdir -p ${mega_path}

curl ${repo_path}/detoker.pt -L --output ${vqgan_path}/detoker.pt
curl ${repo_path}/vocab_mini.json -L --output ${mini_path}/vocab.json
curl ${repo_path}/merges_mini.txt -L --output ${mini_path}/merges.txt
curl ${repo_path}/encoder_mini.pt -L --output ${mini_path}/encoder.pt
curl ${repo_path}/decoder_mini.pt -L --output ${mini_path}/decoder.pt
curl ${repo_path}/vocab.json -L --output ${mega_path}/vocab.json
curl ${repo_path}/merges.txt -L --output ${mega_path}/merges.txt
curl ${repo_path}/encoder.pt -L --output ${mega_path}/encoder.pt
curl ${repo_path}/decoder.pt -L --output ${mega_path}/decoder.pt