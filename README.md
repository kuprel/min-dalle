# min(DALL·E)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb) &nbsp;
[![Replicate](https://replicate.com/kuprel/min-dalle/badge)](https://replicate.com/kuprel/min-dalle)

This is a minimal implementation of Boris Dayma's [DALL·E Mini](https://github.com/borisdayma/dalle-mini) in PyTorch.  It has been stripped to the bare essentials necessary for doing inference.  The only third party dependencies are numpy and torch.

It currently takes **7.4 seconds** to generate an image with DALL·E Mega with PyTorch on a standard GPU runtime in Colab

The flax model, and the code for coverting it to torch, have been moved [here](https://github.com/kuprel/min-dalle-flax).

### Install

```
$ pip install min-dalle
```  

### Usage

Use the python script `image_from_text.py` to generate images from the command line.

To load a model once and generate multiple times, initialize `MinDalleTorch`, then call `generate_image` with some text and a seed.

```
from min_dalle import MinDalleTorch

model = MinDalleTorch(
    is_mega=True, 
    is_reusable=True,
    models_root='./pretrained'
)

image = model.generate_image("court sketch of godzilla on trial", seed=40)
```

Model parameters will be downloaded as needed to the directory specified.  The models can also be manually downloaded [here](https://huggingface.co/kuprel/min-dalle/tree/main).

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
![Godzilla Trial](examples/godzilla_on_trial.png)


```
python image_from_text.py --text='trail cam footage of gollum eating watermelon' --mega --seed=1
```
![Gollum Trailcam](examples/gollum_trailcam.png)