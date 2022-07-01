import jax, flax
from jax import lax, numpy as jnp
from flax import linen as nn
from typing import Tuple

from .dalle_bart_encoder_flax import GLUFlax, AttentionFlax


class DecoderCrossAttentionFlax(AttentionFlax):
    def __call__(
        self,
        decoder_state: jnp.ndarray,
        encoder_state: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        keys = self.k_proj(encoder_state)
        values = self.v_proj(encoder_state)
        queries = self.q_proj(decoder_state)
        return self.forward(keys, values, queries, attention_mask)


class DecoderSelfAttentionFlax(AttentionFlax):
    def __call__(
        self,
        decoder_state: jnp.ndarray,
        attention_state: jnp.ndarray,
        attention_mask: jnp.ndarray,
        state_index: tuple
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        keys = self.k_proj(decoder_state)
        values = self.v_proj(decoder_state)
        queries = self.q_proj(decoder_state)

        attention_state = lax.dynamic_update_slice(
            attention_state, 
            jnp.concatenate([keys, values]).astype(jnp.float32),
            state_index
        )
        batch_count = decoder_state.shape[0]
        keys = attention_state[:batch_count]
        values = attention_state[batch_count:]

        decoder_state = self.forward(
            keys, 
            values,
            queries, 
            attention_mask
        ).astype(decoder_state.dtype)
        return decoder_state, attention_state


class DalleBartDecoderLayerFlax(nn.Module):
    image_token_count: int
    attention_head_count: int
    embed_count: int
    glu_embed_count: int

    def setup(self):
        self.pre_self_attn_layer_norm = nn.LayerNorm(use_scale=False)
        self.self_attn = DecoderSelfAttentionFlax(
            self.attention_head_count,
            self.embed_count
        )
        self.self_attn_layer_norm = nn.LayerNorm()
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(use_scale=False)
        self.encoder_attn = DecoderCrossAttentionFlax(
            self.attention_head_count,
            self.embed_count,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm()
        self.glu = GLUFlax(self.embed_count, self.glu_embed_count)

    @nn.compact
    def __call__(
        self,
        decoder_state: jnp.ndarray,
        encoder_state: jnp.ndarray,
        attention_state: jnp.ndarray,
        attention_mask: jnp.ndarray,
        token_index: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Self Attention
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm(decoder_state)
        self_attention_mask = jnp.tile(
            jnp.arange(self.image_token_count) < token_index + 1, 
            (decoder_state.shape[0], 1)
        )
        decoder_state, attention_state = self.self_attn(
            decoder_state,
            attention_state,
            self_attention_mask,
            (0, token_index, 0)
        )
        decoder_state = self.self_attn_layer_norm(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm(decoder_state)
        decoder_state = self.encoder_attn(
            decoder_state,
            encoder_state,
            attention_mask
        )
        decoder_state = self.encoder_attn_layer_norm(decoder_state)
        decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


@flax.struct.dataclass
class SampleState:
    prev_token: jnp.ndarray
    prng_key: jnp.ndarray
    attention_state: jnp.ndarray

def super_conditioned(logits: jnp.ndarray, a: float) -> jnp.ndarray:
    return (1 - a) * logits[0, -1] + a * logits[1, -1]

def keep_top_k(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    top_logits, _ = lax.top_k(logits, k)
    suppressed = -jnp.inf * jnp.ones_like(logits)
    return lax.select(logits < top_logits[-1], suppressed, logits)

class DalleBartDecoderFlax(nn.Module):
    image_token_count: int
    image_vocab_count: int
    attention_head_count: int
    embed_count: int
    glu_embed_count: int
    layer_count: int
    start_token: int

    def setup(self):
        self.embed_tokens = nn.Embed(
            self.image_vocab_count + 1,
            self.embed_count
        )
        self.embed_positions = nn.Embed(
            self.image_token_count,
            self.embed_count
        )
        self.layers = nn.scan(
            DalleBartDecoderLayerFlax,
            variable_axes = { "params": 0 },
            split_rngs = { "params": True },
            in_axes = (nn.broadcast, 0, nn.broadcast, nn.broadcast),
            out_axes = 0,
            length=self.layer_count,
        )(
            self.image_token_count,
            self.attention_head_count,
            self.embed_count,
            self.glu_embed_count, 
            name="FlaxBartDecoderLayers"
        )
        self.layernorm_embedding = nn.LayerNorm()
        self.final_ln = nn.LayerNorm(use_scale=False)
        self.lm_head = nn.Dense(self.image_vocab_count + 1, use_bias=False)

    def __call__(
        self,
        encoder_state: jnp.ndarray,
        attention_state: jnp.ndarray,
        attention_mask: jnp.ndarray,
        prev_token: int,
        token_index: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_count = encoder_state.shape[0]
        ones = jnp.ones((batch_count, 1), dtype=jnp.int32)
        decoder_state = self.embed_tokens(prev_token * ones) 
        decoder_state += self.embed_positions(token_index * ones)
        decoder_state = self.layernorm_embedding(decoder_state)
        decoder_state, attention_state = self.layers(
            decoder_state,
            encoder_state,
            attention_state,
            attention_mask,
            token_index
        )
        decoder_state = self.final_ln(decoder_state)
        decoder_state = self.lm_head(decoder_state)
        return decoder_state, attention_state

    def sample_image_tokens(
        self,
        text_tokens: jnp.ndarray,
        encoder_state: jnp.ndarray,
        prng_key: jax.random.PRNGKey,
        params: dict
    ) -> jnp.ndarray:
        attention_mask = jnp.not_equal(text_tokens, 1)
        
        def sample_next_image_token(
            state: SampleState,
            token_index: int
        ) -> Tuple[SampleState, jnp.ndarray]:
            logits, attention_state = self.apply(
                { 'params': params },
                encoder_state = encoder_state,
                attention_state = state.attention_state,
                attention_mask = attention_mask,
                prev_token = state.prev_token,
                token_index = token_index
            )
            
            logits = super_conditioned(logits, 10.0)
            logits = keep_top_k(logits, k=50)

            prng_key, prng_key_next = jax.random.split(state.prng_key)
            next_token = jax.random.categorical(prng_key, logits, axis=-1)

            state = SampleState(
                prev_token = next_token,
                prng_key = prng_key_next,
                attention_state = attention_state
            )

            return state, next_token

        batch_count = encoder_state.shape[0]
        attention_state_shape = (
            self.layer_count, 
            batch_count * 2, 
            self.image_token_count, 
            self.embed_count
        )

        initial_state = SampleState(
            prev_token = self.start_token,
            prng_key = prng_key,
            attention_state = jnp.zeros(attention_state_shape)
        )

        _, image_tokens = lax.scan(
            sample_next_image_token,
            initial_state,
            jnp.arange(self.image_token_count)
        )       

        return image_tokens