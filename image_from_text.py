import argparse
import os
import json
import numpy
from PIL import Image
from typing import Tuple, List

from min_dalle.load_params import load_dalle_bart_flax_params
from min_dalle.text_tokenizer import TextTokenizer
from min_dalle.min_dalle_flax import generate_image_tokens_flax
from min_dalle.min_dalle_torch import (
    generate_image_tokens_torch,
    detokenize_torch
)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--text',
    help='text to generate image from',
    type=str
)
parser.add_argument(
    '--seed',
    help='random seed',
    type=int,
    default=0
)
parser.add_argument(
    '--mega',
    help='use larger dalle mega model',
    action=argparse.BooleanOptionalAction
)
parser.add_argument(
    '--torch',
    help='use torch transformers',
    action=argparse.BooleanOptionalAction
)
parser.add_argument(
    '--image_path',
    help='path to save generated image',
    type=str,
    default='generated.png'
)
parser.add_argument(
    '--image_token_count',
    help='number of image tokens to generate (for debugging)',
    type=int,
    default=256
)


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


def tokenize_text(
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


if __name__ == '__main__':
    args = parser.parse_args()

    model_name = 'mega' if args.mega == True else 'mini'
    model_path = './pretrained/dalle_bart_{}'.format(model_name)
    config, vocab, merges = load_dalle_bart_metadata(model_path)
    text_tokens = tokenize_text(args.text, config, vocab, merges)
    params_dalle_bart = load_dalle_bart_flax_params(model_path)

    image_tokens = numpy.zeros(config['image_length'])
    if args.torch == True:
        image_tokens[:args.image_token_count] = generate_image_tokens_torch(
            text_tokens = text_tokens,
            seed = args.seed,
            config = config,
            params = params_dalle_bart,
            image_token_count = args.image_token_count
        )
    else:
        image_tokens[...] = generate_image_tokens_flax(
            text_tokens = text_tokens, 
            seed = args.seed,
            config = config,
            params = params_dalle_bart,
        )

    if args.image_token_count == config['image_length']:
        image = detokenize_torch(image_tokens)
        image = save_image(image, args.image_path)
        print(ascii_from_image(image, size=128))