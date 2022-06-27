import os
import json
import numpy
import torch
from PIL import Image
from typing import Tuple, List

from text_tokenizer import TextTokenizer
from models.vqgan_detokenizer import VQGanDetokenizer
from load_params import load_vqgan_torch_params


def load_dalle_bart_metadata(path: str) -> Tuple[dict, dict, List[str]]:
    print("loading model")
    for f in ['config.json', 'flax_model.msgpack', 'vocab.json', 'merges.txt']:
        assert(os.path.exists(os.path.join(path, f)))
    with open(path + '/config.json', 'r') as f: 
        config = json.load(f)
    with open(path + '/vocab.json') as f:
        vocab = json.load(f)
    with open(path + '/merges.txt') as f:
        merges = f.read().split("\n")[1:-1]
    return config, vocab, merges


def ascii_from_image(image: Image.Image, size: int) -> str:
    rgb_pixels = image.resize((size, int(0.55 * size))).convert('L').getdata()
    chars = list('.,;/IOX')
    chars = [chars[i * len(chars) // 256] for i in rgb_pixels]
    chars = [chars[i * size: (i + 1) * size] for i in range(size // 2)]
    return '\n'.join(''.join(row) for row in chars)


def save_image(image: numpy.ndarray, path: str) -> Image.Image:
    if os.path.isdir(path):
        path = os.path.join(path, 'generated.png')
    elif not path.endswith('.png'):
        path += '.png'
    print("saving image to", path)
    image: Image.Image = Image.fromarray(numpy.asarray(image))
    image.save(path)
    return image


def tokenize(
    text: str, 
    config: dict,
    vocab: dict,
    merges: List[str]
) -> numpy.ndarray:
    print("tokenizing text")
    tokens = TextTokenizer(vocab, merges)(text)
    print("text tokens", tokens)
    text_tokens = numpy.ones((2, config['max_text_length']), dtype=numpy.int32)
    text_tokens[0, :len(tokens)] = tokens
    text_tokens[1, :2] = [tokens[0], tokens[-1]]
    return text_tokens


def detokenize_torch(
    image_tokens: numpy.ndarray, 
    model_path: str
) -> numpy.ndarray:
    print("detokenizing image")
    params = load_vqgan_torch_params(model_path)
    detokenizer = VQGanDetokenizer()
    detokenizer.load_state_dict(params)
    image_tokens = torch.tensor(image_tokens).to(torch.long)
    image = detokenizer.forward(image_tokens).to(torch.uint8)
    return image.detach().numpy()