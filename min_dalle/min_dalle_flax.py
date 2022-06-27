import jax
from jax import numpy as jnp
import numpy

from .models.dalle_bart_encoder_flax import DalleBartEncoderFlax
from .models.dalle_bart_decoder_flax import DalleBartDecoderFlax


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
    text_tokens: numpy.ndarray,
    seed: int,
    config: dict,
    params: dict
) -> numpy.ndarray:
    encoder_state = encode_flax(
        text_tokens, 
        config, 
        params
    )
    image_tokens = decode_flax(
        text_tokens, 
        encoder_state, 
        config, 
        seed, 
        params
    )
    image_tokens = numpy.array(image_tokens)
    print("image tokens", list(image_tokens))
    return image_tokens