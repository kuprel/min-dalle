# min(DALL·E)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb)
&nbsp;
[![Replicate](https://replicate.com/kuprel/min-dalle/badge)](https://replicate.com/kuprel/min-dalle)
&nbsp;
[![Join us on Discord](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.gg/xBPBXfcFHd)

This is a fast, minimal implementation of Boris Dayma's [DALL·E Mega](https://github.com/borisdayma/dalle-mini).  It has been stripped down for inference and converted to PyTorch.  The only third party dependencies are numpy, requests, pillow and torch.

To generate a 4x4 grid of DALL·E Mega images it takes:
- 89 sec with a T4 in Colab
- 48 sec with a P100 in Colab
- 14 sec with an A100 on Replicate
- TBD with an H100 (@NVIDIA?)

The flax model and code for converting it to torch can be found [here](https://github.com/kuprel/min-dalle-flax).

## Install

```bash
$ pip install min-dalle
```  

## Usage

Load the model parameters once and reuse the model to generate multiple images.

```python
from min_dalle import MinDalle

model = MinDalle(
    is_mega=True, 
    is_reusable=True,
    models_root='./pretrained'
)
```

The required models will be downloaded to `models_root` if they are not already there.  Once everything has finished initializing, call `generate_image` with some text as many times as you want.  Use a positive `seed` for reproducible results.  Higher values for `log2_supercondition_factor` result in better agreement with the text but a narrower variety of generated images.

```python
image = model.generate_image(
    'Dali painting of WALL·E', 
    seed=-1,
    grid_size=4,
    log2_supercondition_factor=3
)

display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/dali_walle.jpg" alt="min-dalle" width="300"/>

### Interactive

If the model is being used interactively (e.g. in a notebook) `generate_image_stream` can be used to generate a stream of images as the model is decoding.  The detokenizer adds a slight delay for each image.  Setting `log2_mid_count` to 3 results in a total of `2 ** 3 = 8` generated images.  The only valid values for `log2_mid_count` are 0, 1, 2, 3, 4.

```python
image_stream = model.generate_image_stream(
    text='Dali painting of WALL·E',
    seed=-1,
    grid_size=3,
    log2_mid_count=3,
    log2_supercondition_factor=3
)

for image in image_stream:
    display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/dali_walle_animated.gif" alt="min-dalle" width="300"/>

### Command Line

Use `image_from_text.py` to generate images from the command line.

```bash
$ python image_from_text.py --text='artificial intelligence' --no-mega
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/artificial_intelligence.jpg" alt="min-dalle" width="200"/>
