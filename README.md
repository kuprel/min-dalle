# min(DALL·E)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb) &nbsp;
[![Replicate](https://replicate.com/kuprel/min-dalle/badge)](https://replicate.com/kuprel/min-dalle)

This is a minimal implementation of Boris Dayma's [DALL·E Mini](https://github.com/borisdayma/dalle-mini) in PyTorch.  It has been stripped to the bare essentials necessary for doing inference.  The only third party dependencies are numpy and torch.

It currently takes **7.4 seconds** to generate an image with DALL·E Mega with PyTorch on a standard GPU runtime in Colab

The flax model, and the code for coverting it to torch, has been moved [here](https://github.com/kuprel/min-dalle-flax).

### Setup

Run `sh setup.sh` to install dependencies and download pretrained models.  The torch models can be manually downloaded [here](https://huggingface.co/kuprel/min-dalle/tree/main).

### Usage

Use the python script `image_from_text.py` to generate images from the command line.  Note: the command line script loads the models and parameters each time.  To load a model once and generate multiple times, initialize `MinDalleTorch`, then call `generate_image` with some text and a seed.  See the colab for an example.

### Examples

```
python image_from_text.py --text='artificial intelligence' --seed=7
```
![Alien](examples/artificial_intelligence.png)


```
python image_from_text.py --text='a comfy chair that looks like an avocado' --mega --seed=10
```
![Avocado Armchair](examples/avocado_armchair.png)


```
python image_from_text.py --text='court sketch of godzilla on trial' --mega --seed=40
```

![Godzilla Trial](examples/godzilla_trial.png)


```
python image_from_text.py --text='trail cam footage of gollum eating watermelon' --mega --seed=1
```
![Gollum Trailcam](examples/gollum_trailcam.png)