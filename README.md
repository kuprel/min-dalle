# min(DALL·E)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb)
&nbsp;
[![Replicate](https://replicate.com/kuprel/min-dalle/badge)](https://replicate.com/kuprel/min-dalle)
&nbsp;
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Demo-blue)](https://huggingface.co/spaces/kuprel/min-dalle)
&nbsp;
[![Discord](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.com/channels/823813159592001537/912729332311556136)

[YouTube Walk-through](https://youtu.be/x_8uHX5KngE) by The AI Epiphany

This is a fast, minimal port of Boris Dayma's [DALL·E Mini](https://github.com/borisdayma/dalle-mini) (with mega weights).  It has been stripped down for inference and converted to PyTorch.  The only third party dependencies are numpy, requests, pillow and torch.

To generate a 4x4 grid of DALL·E Mega images it takes:
- 89 sec with a T4 in Colab
- 48 sec with a P100 in Colab
- 13 sec with an A100 on Replicate

Here's a more detailed breakdown of performance on an A100. Credit to [@technobird22](https://github.com/technobird22) and his [NeoGen](https://github.com/technobird22/NeoGen) discord bot for the graph.
<br />
<img src="https://github.com/kuprel/min-dalle/raw/main/performance.png" alt="min-dalle" width="450"/>
<br />

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
    device='cuda',
    is_mega=True, 
    is_reusable=True
)
```

The required models will be downloaded to `models_root` if they are not already there.  Set the `dtype` to `torch.float16` to save GPU memory.  If you have an Ampere architecture GPU you can use `torch.bfloat16`.  Set the `device` to either "cuda" or "cpu".  Once everything has finished initializing, call `generate_image` with some text as many times as you want.  Use a positive `seed` for reproducible results.  Higher values for `supercondition_factor` result in better agreement with the text but a narrower variety of generated images.  Every image token is sampled from the `top_k` most probable tokens.  The largest logit is subtracted from the logits to avoid infs.  The logits are then divided by the `temperature`.  If `is_seamless` is true, the image grid will be tiled in token space not pixel space.

```python
image = model.generate_image(
    text='Nuclear explosion broccoli',
    seed=-1,
    grid_size=4,
    is_seamless=False,
    temperature=1,
    top_k=256,
    supercondition_factor=32,
    is_verbose=False
)

display(image)
```
<img src="https://github.com/kuprel/min-dalle/raw/main/examples/nuclear_broccoli.jpg" alt="min-dalle" width="400"/>

Credit to [@hardmaru](https://twitter.com/hardmaru) for the [example](https://twitter.com/hardmaru/status/1544354119527596034)


### Saving Individual Images
The images can also be generated as a `FloatTensor` in case you want to process them manually.

```python
images = model.generate_images(
    text='Nuclear explosion broccoli',
    seed=-1,
    grid_size=3,
    is_seamless=False,
    temperature=1,
    top_k=256,
    supercondition_factor=16,
    is_verbose=False
)
```

To get an image into PIL format you will have to first move the images to the CPU and convert the tensor to a numpy array.
```python
images = images.to('cpu').numpy()
```
Then image $i$ can be coverted to a PIL.Image and saved
```python
image = Image.fromarray(images[i])
image.save('image_{}.png'.format(i))
```

### Progressive Outputs

If the model is being used interactively (e.g. in a notebook) `generate_image_stream` can be used to generate a stream of images as the model is decoding.  The detokenizer adds a slight delay for each image.  Set `progressive_outputs` to `True` to enable this.  An example is implemented in the colab.

```python
image_stream = model.generate_image_stream(
    text='Dali painting of WALL·E',
    seed=-1,
    grid_size=3,
    progressive_outputs=True,
    is_seamless=False,
    temperature=1,
    top_k=256,
    supercondition_factor=16,
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
