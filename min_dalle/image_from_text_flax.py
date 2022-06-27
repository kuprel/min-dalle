import jax
from jax import numpy as jnp
import numpy
import argparse

from load_params import load_dalle_bart_flax_params
from image_from_text import (
    load_dalle_bart_metadata, 
    tokenize, 
    detokenize_torch,
    save_image, 
    ascii_from_image
)
from models.dalle_bart_encoder_flax import DalleBartEncoderFlax
from models.dalle_bart_decoder_flax import DalleBartDecoderFlax


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


def encode_flax(
    text_tokens: numpy.ndarray,
    config: dict, 
    params: dict
) -> jnp.ndarray:
    print("loading flax encoder")
    encoder: DalleBartEncoderFlax = DalleBartEncoderFlax(
        attention_head_count = config['encoder_attention_heads'],
        embed_count = config['d_model'],
        glu_embed_count = config['encoder_ffn_dim'],
        text_token_count = config['max_text_length'],
        text_vocab_count = config['encoder_vocab_size'],
        layer_count = config['encoder_layers']
    ).bind({'params': params.pop('encoder')})

    print("encoding text tokens")
    encoder_state = encoder(text_tokens)
    del encoder
    return encoder_state

def decode_flax(
    text_tokens: jnp.ndarray,
    encoder_state: jnp.ndarray,
    config: dict,
    seed: int,
    params: dict
) -> jnp.ndarray:
    print("loading flax decoder")
    decoder = DalleBartDecoderFlax(
        image_token_count = config['image_length'],
        text_token_count = config['max_text_length'],
        image_vocab_count = config['image_vocab_size'],
        attention_head_count = config['decoder_attention_heads'],
        embed_count = config['d_model'],
        glu_embed_count = config['decoder_ffn_dim'],
        layer_count = config['decoder_layers'],
        start_token = config['decoder_start_token_id']
    )
    print("sampling image tokens")
    image_tokens = decoder.sample_image_tokens(
        text_tokens,
        encoder_state,
        jax.random.PRNGKey(seed),
        params.pop('decoder')
    )
    del decoder
    return image_tokens

def generate_image_tokens_flax(
    text: str,
    seed: int, 
    dalle_bart_path: str
) -> numpy.ndarray:
    config, vocab, merges = load_dalle_bart_metadata(dalle_bart_path)
    text_tokens = tokenize(text, config, vocab, merges)
    params_dalle_bart = load_dalle_bart_flax_params(dalle_bart_path)
    encoder_state = encode_flax(text_tokens, config, params_dalle_bart)
    image_tokens = decode_flax(
        text_tokens, 
        encoder_state, 
        config, seed, 
        params_dalle_bart
    )
    return numpy.array(image_tokens)

if __name__ == '__main__':
    args = parser.parse_args()

    image_tokens = generate_image_tokens_flax(
        args.text, 
        args.seed, 
        args.dalle_bart_path
    )
    print("image tokens", list(image_tokens))
    image = detokenize_torch(image_tokens, args.vqgan_path)
    image = save_image(image, args.image_path)
    print(ascii_from_image(image, size=128))