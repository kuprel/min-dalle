# min(DALL·E)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb)
&nbsp;
[![Replicate](https://replicate.com/kuprel/min-dalle/badge)](https://replicate.com/kuprel/min-dalle)
&nbsp;
[![Join us on Discord](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.gg/xBPBXfcFHd)

This is a minimal implementation of Boris Dayma's [DALL·E Mini](https://github.com/borisdayma/dalle-mini) in PyTorch.  It has been stripped to the bare essentials necessary for doing inference.  The only third party dependencies are numpy and torch.

It currently take **35 seconds** to generate a 3x3 grid with DALL·E Mega on a standard GPU runtime in Colab.

The flax model and code for converting it to torch can be found [here](https://github.com/kuprel/min-dalle-flax).

## Install

```bash
$ pip install min-dalle
```  

## Usage

### Python

Load the model parameters once and reuse the model to generate multiple images.

```python
from min_dalle import MinDalle

model = MinDalle(is_mega=True, models_root='./pretrained')
```

The required models will be downloaded to `models_root` if they are not already there.  Once everything has finished initializing, call `generate_image` with some text and a seed as many times as you want.

```python
text = "a comfy chair that looks like an avocado"
image = model.generate_image(text)
display(image)
```
![Avocado Armchair](https://github.com/kuprel/min-dalle/raw/main/examples/avocado_armchair.png)

```python
text = "trail cam footage of gollum eating watermelon"
image = model.generate_image(text, seed=1, grid_size=3)
display(image)
```
![Gollum Trailcam](https://github.com/kuprel/min-dalle/raw/main/examples/gollum_trailcam.png)


### Command Line

Use `image_from_text.py` to generate images from the command line.

```bash
$ python image_from_text.py --text='artificial intelligence' --seed=7
```
![Artificial Intelligence](https://github.com/kuprel/min-dalle/raw/main/examples/artificial_intelligence.png)

```bash
$ python image_from_text.py --text='court sketch of godzilla on trial' --mega
```
![Godzilla Trial](https://github.com/kuprel/min-dalle/raw/main/examples/godzilla_on_trial.png)