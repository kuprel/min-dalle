import numpy
import torch
from torch import Tensor
import argparse
from typing import Dict

from min_dalle.image_from_text import (
    load_dalle_bart_metadata, 
    tokenize,
    detokenize_torch,
    save_image, 
    ascii_from_image
)
from min_dalle.models.dalle_bart_encoder_torch import DalleBartEncoderTorch
from min_dalle.models.dalle_bart_decoder_torch import DalleBartDecoderTorch

from min_dalle.load_params import (
    load_dalle_bart_flax_params, 
    convert_dalle_bart_torch_from_flax_params
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
    '--image_token_count',
    help='image tokens to sample',
    type=int,
    default=256
)
parser.add_argument(
    '--image_path',
    help='generated image path',
    type=str,
    default='generated.png'
)
parser.add_argument(
    '--dalle_bart_path',
    help='pretraied dalle bart path',
    type=str,
    default='./pretrained/dalle_bart_mini'
)
parser.add_argument(
    '--vqgan_path',
    help='pretraied vqgan path',
    type=str,
    default='./pretrained/vqgan'
)


def encode_torch(
    text_tokens: numpy.ndarray,
    config: dict, 
    params: dict
) -> Tensor:
    print("loading torch encoder")
    encoder = DalleBartEncoderTorch(
        layer_count = config['encoder_layers'],
        embed_count = config['d_model'],
        attention_head_count = config['encoder_attention_heads'],
        text_vocab_count = config['encoder_vocab_size'],
        text_token_count = config['max_text_length'],
        glu_embed_count = config['encoder_ffn_dim']
    )
    encoder_params = convert_dalle_bart_torch_from_flax_params(
        params.pop('encoder'), 
        layer_count=config['encoder_layers'], 
        is_encoder=True
    )
    encoder.load_state_dict(encoder_params, strict=False)
    del encoder_params

    print("encoding text tokens")
    text_tokens = torch.tensor(text_tokens).to(torch.long)
    encoder_state = encoder(text_tokens)
    del encoder
    return encoder_state


def decode_torch(
    text_tokens: Tensor,
    encoder_state: Tensor, 
    config: dict,
    seed: int,
    params: dict,
    image_token_count: int
) -> Tensor:
    print("loading torch decoder")
    decoder = DalleBartDecoderTorch(
        image_vocab_size = config['image_vocab_size'],
        image_token_count = config['image_length'],
        sample_token_count = image_token_count,
        embed_count = config['d_model'],
        attention_head_count = config['decoder_attention_heads'],
        glu_embed_count = config['decoder_ffn_dim'],
        layer_count = config['decoder_layers'],
        batch_count = 2,
        start_token = config['decoder_start_token_id'],
        is_verbose = True
    )
    decoder_params = convert_dalle_bart_torch_from_flax_params(
        params.pop('decoder'), 
        layer_count=config['decoder_layers'],
        is_encoder=False
    )
    decoder.load_state_dict(decoder_params, strict=False)
    del decoder_params

    print("sampling image tokens")
    torch.manual_seed(seed)
    text_tokens = torch.tensor(text_tokens).to(torch.long)
    image_tokens = decoder.forward(text_tokens, encoder_state)
    return image_tokens


def generate_image_tokens_torch(
    text: str, 
    seed: int, 
    image_token_count: int,
    dalle_bart_path: str
) -> numpy.ndarray:
    config, vocab, merges = load_dalle_bart_metadata(dalle_bart_path)
    text_tokens = tokenize(text, config, vocab, merges)
    params_dalle_bart = load_dalle_bart_flax_params(dalle_bart_path)
    encoder_state = encode_torch(text_tokens, config, params_dalle_bart)
    image_tokens = decode_torch(
        text_tokens, 
        encoder_state, 
        config, seed, params_dalle_bart,
        image_token_count
    )
    return image_tokens.detach().numpy()


if __name__ == '__main__':
    args = parser.parse_args()
    image_tokens = generate_image_tokens_torch(
        args.text, 
        args.seed, 
        args.image_token_count, 
        args.dalle_bart_path
    )
    if args.image_token_count < 256:
        print("image tokens", list(image_tokens, ))
    else:
        image = detokenize_torch(image_tokens, args.vqgan_path)
        image = save_image(image, args.image_path)
        print(ascii_from_image(image, size=128))

    