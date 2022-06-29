# min(DALL·E)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb)

This is a minimal implementation of [DALL·E Mini](https://github.com/borisdayma/dalle-mini).  It has been stripped to the bare essentials necessary for doing inference, and converted to PyTorch.  The only third party dependencies are `numpy` and `torch` for the torch model and `flax` for the flax model.

### Setup

Run `sh setup.sh` to install dependencies and download pretrained models.  The models can also be downloaded manually: 
[VQGan](https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384), 
[DALL·E Mini](https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mini-1/v0/files), 
[DALL·E Mega](https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mega-1-fp16/v14/files)

### Usage

Use the command line python script `main.py` to generate images. Here are some examples:

```
python main.py --text='alien life' --seed=7
```
![Alien](examples/alien.png)


```
python main.py --text='a comfy chair that looks like an avocado' --mega --seed=4
```
![Avocado Armchair](examples/avocado_armchair.png)


```
python main.py --text='court sketch of godzilla on trial' --mega --seed=100
```

![Godzilla Trial](examples/godzilla_trial.png)
