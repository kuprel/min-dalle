from functools import partial
import jax
from jax import lax, numpy as jnp
from flax import linen as nn


class GLUFlax(nn.Module):
    count_in_out: int
    count_middle: int

    def setup(self):
        self.gelu = partial(nn.gelu, approximate=False)
        self.ln0 = nn.LayerNorm(use_scale=False)
        self.ln1 = nn.LayerNorm(use_scale=False)
        self.fc0 = nn.Dense(self.count_middle, use_bias=False)
        self.fc1 = nn.Dense(self.count_middle, use_bias=False)
        self.fc2 = nn.Dense(self.count_in_out, use_bias=False)

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = self.ln0(z)
        z = self.ln1(self.gelu(self.fc0(z)) * self.fc1(z))
        z = self.fc2(z)
        return z


class AttentionFlax(nn.Module):
    head_count: int
    embed_count: int

    def setup(self):
        self.q_proj = nn.Dense(self.embed_count, use_bias=False)
        self.k_proj = nn.Dense(self.embed_count, use_bias=False)
        self.v_proj = nn.Dense(self.embed_count, use_bias=False)
        self.out_proj = nn.Dense(self.embed_count, use_bias=False)

    def forward(
        self,
        keys: jnp.ndarray,
        values: jnp.ndarray,
        queries: jnp.ndarray,
        attention_mask: jnp.ndarray
    ) -> jnp.ndarray:
        keys = keys.reshape(keys.shape[:2] + (self.head_count, -1))
        values = values.reshape(values.shape[:2] + (self.head_count, -1))
        queries = queries.reshape(queries.shape[:2] + (self.head_count, -1))
        queries /= queries.shape[-1] ** 0.5
        attention_bias: jnp.ndarray = lax.select(
            attention_mask,
            jnp.full(attention_mask.shape, 0.0),
            jnp.full(attention_mask.shape, -jnp.inf),
        )
        attention_weights: jnp.ndarray = jnp.einsum(
            'bqhd,bkhd->bhqk', 
            queries, 
            keys
        )
        attention_weights += attention_bias[:, None, None, :]
        attention_weights = jax.nn.softmax(attention_weights)
        attention_output: jnp.ndarray = jnp.einsum(
            "bhqk,bkhd->bqhd", 
            attention_weights, 
            values
        )
        shape = attention_output.shape[:2] + (self.embed_count,)
        attention_output = attention_output.reshape(shape)
        attention_output = self.out_proj(attention_output)
        return attention_output


class EncoderSelfAttentionFlax(AttentionFlax):
    def __call__(
        self,
        encoder_state: jnp.ndarray,
        attention_mask: jnp.ndarray
    ) -> jnp.ndarray:
        keys = self.k_proj(encoder_state)
        values = self.v_proj(encoder_state)
        queries = self.q_proj(encoder_state)
        return self.forward(keys, values, queries, attention_mask)


class DalleBartEncoderLayerFlax(nn.Module):
    attention_head_count: int
    embed_count: int
    glu_embed_count: int

    def setup(self):
        self.pre_self_attn_layer_norm = nn.LayerNorm(use_scale=False)
        self.self_attn = EncoderSelfAttentionFlax(
            self.attention_head_count,
            self.embed_count
        )
        self.self_attn_layer_norm = nn.LayerNorm()
        self.glu = GLUFlax(self.embed_count, self.glu_embed_count)

    @nn.compact
    def __call__(
        self,
        encoder_state: jnp.ndarray,
        attention_mask: jnp.ndarray
    ) -> jnp.ndarray:
        residual = encoder_state
        encoder_state = self.pre_self_attn_layer_norm(encoder_state)
        encoder_state = self.self_attn(encoder_state, attention_mask)
        encoder_state = self.self_attn_layer_norm(encoder_state)
        encoder_state = residual + encoder_state
        residual = encoder_state
        encoder_state = self.glu(encoder_state)
        encoder_state = residual + encoder_state
        return encoder_state, None


class DalleBartEncoderFlax(nn.Module):
    attention_head_count: int
    embed_count: int
    glu_embed_count: int
    text_token_count: int
    text_vocab_count: int
    layer_count: int

    def setup(self):
        self.embed_tokens = nn.Embed(self.text_vocab_count, self.embed_count)
        self.embed_positions = nn.Embed(self.text_token_count, self.embed_count)
        self.layers = nn.scan(
            DalleBartEncoderLayerFlax,
            variable_axes = { "params": 0 },
            split_rngs = { "params": True },
            in_axes = nn.broadcast,
            length = self.layer_count
        )(
            self.attention_head_count,
            self.embed_count,
            self.glu_embed_count, 
            name="FlaxBartEncoderLayers"
        )
        self.layernorm_embedding = nn.LayerNorm()
        self.final_ln = nn.LayerNorm(use_scale=False)

    def __call__(self, text_tokens: jnp.ndarray) -> jnp.ndarray:
        batch_count, token_count = text_tokens.shape
        pose_tokens = jnp.tile(jnp.arange(token_count), (batch_count, 1))
        attention_mask = jnp.not_equal(text_tokens, 1)
        encoder_state = (
            self.embed_tokens(text_tokens) +
            self.embed_positions(pose_tokens)
        )
        encoder_state = self.layernorm_embedding(encoder_state)
        encoder_state, _ = self.layers(encoder_state, attention_mask)
        encoder_state = self.final_ln(encoder_state)
        return encoder_state