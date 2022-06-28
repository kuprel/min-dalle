import sys
import numpy
import torch
from torch import Tensor
from typing import Dict

from .models.vqgan_detokenizer import VQGanDetokenizer
from .models.dalle_bart_encoder_torch import DalleBartEncoderTorch
from .models.dalle_bart_decoder_torch import DalleBartDecoderTorch

from .load_params import (
    load_vqgan_torch_params,
    convert_dalle_bart_torch_from_flax_params
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
    text_tokens: numpy.ndarray,
    seed: int,
    config: dict,
    params: dict,
    image_token_count: int
) -> numpy.ndarray:
    encoder_state = encode_torch(
        text_tokens, 
        config, 
        params
    )
    image_tokens = decode_torch(
        text_tokens, 
        encoder_state, 
        config, 
        seed, 
        params,
        image_token_count
    )
    return image_tokens.detach().numpy()


def detokenize_torch(image_tokens: numpy.ndarray) -> numpy.ndarray:
    print("detokenizing image")
    in_colab = 'google.colab' in sys.modules
    base_path = './content' if in_colab else '.'
    model_path = f'{base_path}/pretrained/vqgan'
    params = load_vqgan_torch_params(model_path)
    detokenizer = VQGanDetokenizer()
    detokenizer.load_state_dict(params)
    image_tokens = torch.tensor(image_tokens).to(torch.long)
    image = detokenizer.forward(image_tokens).to(torch.uint8)
    return image.detach().numpy()
    