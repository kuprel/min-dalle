# min(DALL·E)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb)
&nbsp;
[![Replicate](https://replicate.com/kuprel/min-dalle/badge)](https://replicate.com/kuprel/min-dalle)
&nbsp;
[![Join us on Discord](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.gg/xBPBXfcFHd)

This is a fast, minimal implementation of Boris Dayma's [DALL·E Mega](https://github.com/borisdayma/dalle-mini).  It has been stripped down for inference and converted to PyTorch.  The only third party dependencies are numpy, requests, pillow and torch.

It takes
- **32 seconds** to generate a 3x3 grid with a P100 in Colab
- **16 seconds** to generate a 4x4 grid with an A100 on Replicate
- **TBD** to generate a 4x4 grid with an H100 (@NVIDIA?)

The flax model and code for converting it to torch can be found [here](https://github.com/kuprel/min-dalle-flax).

## Install

```bash
$ pip install min-dalle
```  

## Usage

Load the model parameters once and reuse the model to generate multiple images.

```python
from min_dalle import MinDalle

model = MinDalle(is_mega=True, models_root='./pretrained')
```

The required models will be downloaded to `models_root` if they are not already there.  Once everything has finished initializing, call `generate_image` with some text and a seed as many times as you want.

```python
text = 'Dali painting of WALL·E'
image = model.generate_image(text, seed=0, grid_size=4)
display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/dali_walle.jpg" alt="min-dalle" width="300"/>

```python
text = 'Rusty Iron Man suit found abandoned in the woods being reclaimed by nature'
image = model.generate_image(text, seed=0, grid_size=3)
display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/ironman.jpg" alt="min-dalle" width="300"/>

```python
text = 'court sketch of godzilla on trial'
image = model.generate_image(text, seed=6, grid_size=3)
display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/godzilla_trial.jpg" alt="min-dalle" width="300"/>

```python
text = 'a funeral at Whole Foods'
image = model.generate_image(text, seed=10, grid_size=3)
display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/funeral.jpg" alt="min-dalle" width="300"/>

```python
text = 'Jesus turning water into wine on Americas Got Talent'
image = model.generate_image(text, seed=2, grid_size=3)
display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/jesus.jpg" alt="min-dalle" width="300"/>

```python
text = 'cctv footage of Yoda robbing a liquor store'
image = model.generate_image(text, seed=0, grid_size=3)
display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/yoda.jpg" alt="min-dalle" width="300"/>


### Command Line

Use `image_from_text.py` to generate images from the command line.

```bash
$ python image_from_text.py --text='artificial intelligence' --no-mega --seed=7
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/artificial_intelligence.jpg" alt="min-dalle" width="200"/>

```bash
$ python image_from_text.py --text='trail cam footage of gollum eating watermelon' --mega --seed=1 --grid-size=3
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/gollum_trailcam.jpg" alt="min-dalle" width="300"/>
