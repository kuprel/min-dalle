# min DALL·E

This is a minimal implementation of [DALL·E Mini](https://github.com/borisdayma/dalle-mini) in both Flax and PyTorch

### Setup

Run `sh setup.sh` to install dependencies and download pretrained models.  The only required dependencies are `flax` and `torch`.  In the bash script, GitHub LFS is used to download the VQGan detokenizer and the Weight & Biases python package is used to download the DALL·E Mini and DALL·E Mega transformer models.  You can also download those files manually by visting the links in the bash script.

### Run

Here are some examples

```
python3 image_from_text_flax.py \
  --dalle_bart_path='./pretrained/dalle_bart_mega' \
  --vqgan_path='./pretrained/vqgan' \
  --image_path='./generated/avacado_armchair_flax.png' \
  --seed=4 \
  --text='a comfy chair that looks like an avocado'
```
![Avocado Armchair](examples/avocado_armchair.png)


```
python3 image_from_text_flax.py \
  --dalle_path='./pretrained/dalle-mega' \
  --seed=100 \
  --image_path='./generated/godzilla_trial.png' \
  --text='court sketch of godzilla on trial'
```

![Godzilla Trial](examples/godzilla_trial.png)