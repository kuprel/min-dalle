# min(DALL·E)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb)
&nbsp;
[![Replicate](https://replicate.com/kuprel/min-dalle/badge)](https://replicate.com/kuprel/min-dalle)
&nbsp;
[![Discord](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.com/channels/823813159592001537/912729332311556136)

This is a fast, minimal port of Boris Dayma's [DALL·E Mega](https://github.com/borisdayma/dalle-mini).  It has been stripped down for inference and converted to PyTorch.  The only third party dependencies are numpy, requests, pillow and torch.

To generate a 4x4 grid of DALL·E Mega images it takes:
- 89 sec with a T4 in Colab
- 48 sec with a P100 in Colab
- 14 sec with an A100 on Replicate

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
    models_root='./pretrained',
    dtype=torch.float32,
    is_mega=True, 
    is_reusable=True
)
```

The required models will be downloaded to `models_root` if they are not already there.  If you have an Ampere architecture GPU you can set the `dtype=torch.bfloat16` and save GPU memory.  There is still an issue with `torch.float16` that needs to be sorted out.  Once everything has finished initializing, call `generate_image` with some text as many times as you want.  Use a positive `seed` for reproducible results.  Higher values for `log2_supercondition_factor` result in better agreement with the text but a narrower variety of generated images.  Every image token is sampled from the top-$k$ most probable tokens.

```python
image = model.generate_image(
    text='Nuclear explosion broccoli',
    seed=-1,
    grid_size=4,
    log2_k=6,
    log2_supercondition_factor=5,
    is_verbose=False
)

display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/nuclear_broccoli.jpg" alt="min-dalle" width="400"/>
credit: https://twitter.com/hardmaru/status/1544354119527596034

### Interactive

If the model is being used interactively (e.g. in a notebook) `generate_image_stream` can be used to generate a stream of images as the model is decoding.  The detokenizer adds a slight delay for each image.  Setting `log2_mid_count` to 3 results in a total of `2 ** 3 = 8` generated images.  The only valid values for `log2_mid_count` are 0, 1, 2, 3, and 4.  This is implemented in the colab.

```python
image_stream = model.generate_image_stream(
    text='Dali painting of WALL·E',
    seed=-1,
    grid_size=3,
    log2_mid_count=3,
    log2_k=6,
    log2_supercondition_factor=3,
    is_verbose=False
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
