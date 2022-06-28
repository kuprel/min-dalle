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
        keys: jnp.ndarray = self.k_proj(encoder_state)
        values: jnp.ndarray = self.v_proj(encoder_state)
        queries: jnp.ndarray = self.q_proj(decoder_state)
        query_shape = queries.shape[:2] + (self.head_count, -1)
        key_value_shape = keys.shape[:2] + (self.head_count, -1)
        keys = keys.reshape(key_value_shape)
        values = values.reshape(key_value_shape)
        queries = queries.reshape(query_shape)
        queries /= queries.shape[-1] ** 0.5
        return self.forward(keys, values, queries, attention_mask)


class DecoderSelfAttentionFlax(AttentionFlax):
    def __call__(self,
        decoder_state: jnp.ndarray,
        keys_state: jnp.ndarray,
        values_state: jnp.ndarray,
        attention_mask: jnp.ndarray,
        state_index: tuple
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        shape_split = decoder_state.shape[:2] + (self.head_count, -1)
        keys_state = lax.dynamic_update_slice(
            keys_state, 
            self.k_proj(decoder_state).reshape(shape_split), 
            state_index
        )
        values_state = lax.dynamic_update_slice(
            values_state, 
            self.v_proj(decoder_state).reshape(shape_split), 
            state_index
        )
        queries = self.q_proj(decoder_state).reshape(shape_split)
        queries /= queries.shape[-1] ** 0.5
        decoder_state = self.forward(
            keys_state, 
            values_state, 
            queries, 
            attention_mask
        )
        return decoder_state, (keys_state, values_state)


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
    def __call__(self,
        decoder_state: jnp.ndarray,
        encoder_state: jnp.ndarray,
        keys_state: jnp.ndarray,
        values_state: jnp.ndarray,
        attention_mask: jnp.ndarray,
        token_index: int
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        # Self Attention
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm(decoder_state)
        self_attention_mask = jnp.tile(
            jnp.arange(self.image_token_count) < token_index + 1, 
            (decoder_state.shape[0], 1)
        )
        decoder_state, keys_values_state = self.self_attn(
            decoder_state,
            keys_state,
            values_state,
            self_attention_mask,
            (0, token_index, 0, 0)
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

        return decoder_state, keys_values_state


@flax.struct.dataclass
class SampleState:
    prev_token: jnp.ndarray
    prng_key: jnp.ndarray
    keys_state: jnp.ndarray
    values_state: jnp.ndarray

def super_conditioned(logits: jnp.ndarray, a: float) -> jnp.ndarray:
    return a * logits[0, -1] + (1 - a) * logits[1, -1]

def keep_top_k(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    top_logits, _ = lax.top_k(logits, k)
    suppressed = -jnp.inf * jnp.ones_like(logits)
    return lax.select(logits < top_logits[-1], suppressed, logits)

class DalleBartDecoderFlax(nn.Module):
    image_token_count: int
    text_token_count: int
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
            variable_axes = { "params": 0, "cache": 0 },
            split_rngs = { "params": True },
            in_axes = (nn.broadcast, 0, 0, nn.broadcast, nn.broadcast),
            out_axes = (0, 0),
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

    def __call__(self,
        encoder_state: jnp.ndarray,
        keys_state: jnp.ndarray,
        values_state: jnp.ndarray,
        attention_mask: jnp.ndarray,
        prev_token: int,
        token_index: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch_count = encoder_state.shape[0]
        ones = jnp.ones((batch_count, 1), dtype=jnp.int32)
        decoder_state = self.embed_tokens(prev_token * ones) 
        decoder_state += self.embed_positions(token_index * ones)
        decoder_state = self.layernorm_embedding(decoder_state)
        decoder_state, (keys_state, values_state) = self.layers(
            decoder_state,
            encoder_state,
            keys_state,
            values_state,
            attention_mask,
            token_index
        )
        decoder_state = self.final_ln(decoder_state)
        decoder_state = self.lm_head(decoder_state)
        return decoder_state, keys_state, values_state

    def sample_image_tokens(self,
        text_tokens: jnp.ndarray,
        encoder_state: jnp.ndarray,
        prng_key: jax.random.PRNGKey,
        params: dict
    ) -> jnp.ndarray:
        attention_mask = jnp.not_equal(text_tokens, 1)
        
        def sample_next_image_token(
            state: SampleState,
            token_index: int
        ) -> Tuple[SampleState, None]:
            logits, keys_state, values_state = self.apply(
                { 'params': params },
                encoder_state = encoder_state,
                keys_state = state.keys_state,
                values_state = state.values_state,
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
                keys_state = keys_state,
                values_state = values_state
            )

            return state, next_token

        batch_count = encoder_state.shape[0]
        state_shape = (
            self.layer_count, 
            batch_count, 
            self.image_token_count, 
            self.attention_head_count, 
            self.embed_count // self.attention_head_count
        )

        initial_state = SampleState(
            prev_token = self.start_token,
            prng_key = prng_key,
            keys_state = jnp.zeros(state_shape),
            values_state = jnp.zeros(state_shape)
        )

        _, image_tokens = lax.scan(
            sample_next_image_token,
            initial_state,
            jnp.arange(self.image_token_count)
        )       

        return image_tokens