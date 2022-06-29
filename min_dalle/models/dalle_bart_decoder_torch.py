from typing import List, Tuple
import torch
from torch import LongTensor, nn, FloatTensor, BoolTensor
torch.set_grad_enabled(False)

from .dalle_bart_encoder_torch import GLUTorch, AttentionTorch


class DecoderCrossAttentionTorch(AttentionTorch):
    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        query_shape = queries.shape[:2] + (self.head_count, -1)
        key_value_shape = keys.shape[:2] + (self.head_count, -1)
        keys = keys.reshape(key_value_shape)
        values = values.reshape(key_value_shape)
        queries = queries.reshape(query_shape)
        queries /= queries.shape[-1] ** 0.5
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttentionTorch(AttentionTorch):
    def forward(self, 
        decoder_state: FloatTensor,
        keys_values: FloatTensor,
        attention_mask: BoolTensor,
        token_mask: BoolTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        batch_count = decoder_state.shape[0]
        shape = (batch_count, 1) + keys_values.shape[2:]
        keys = self.k_proj.forward(decoder_state).view(shape)
        values = self.v_proj.forward(decoder_state).view(shape)
        keys_values = torch.where(
            token_mask[None, :, None, None], 
            torch.cat([keys, values]), 
            keys_values
        )
        queries = self.q_proj.forward(decoder_state).reshape(shape)
        queries /= queries.shape[-1] ** 0.5
        keys, values = keys_values[:batch_count], keys_values[batch_count:]
        decoder_state = super().forward(keys, values, queries, attention_mask)
        return decoder_state, keys_values


class DecoderLayerTorch(nn.Module):
    def __init__(self, 
        image_token_count: int,
        head_count: int, 
        embed_count: int,
        glu_embed_count: int
    ):
        super().__init__()
        self.image_token_count = image_token_count
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = DecoderSelfAttentionTorch(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.encoder_attn = DecoderCrossAttentionTorch(head_count, embed_count)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLUTorch(embed_count, glu_embed_count)

        self.token_indices = torch.arange(self.image_token_count)
        if torch.cuda.is_available():
            self.token_indices = self.token_indices.cuda()

    def forward(self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        keys_values_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        self_attn_mask = self.token_indices < token_index + 1
        token_mask = self.token_indices == token_index
        self_attn_mask = torch.stack([self_attn_mask] * decoder_state.shape[0])
        decoder_state, keys_values_state = self.self_attn.forward(
            decoder_state,
            keys_values_state,
            self_attn_mask,
            token_mask
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state,
            encoder_state,
            attention_mask
        )
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, keys_values_state


class DalleBartDecoderTorch(nn.Module):
    def __init__(self,
        image_vocab_size: int,
        image_token_count: int,
        sample_token_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        batch_count: int,
        start_token: int,
        is_verbose: bool
    ):
        super().__init__()
        self.is_verbose = is_verbose
        self.layer_count = layer_count
        self.sample_token_count = sample_token_count
        self.condition_factor = 10.0
        self.image_token_count = image_token_count
        self.embed_tokens = nn.Embedding(image_vocab_size + 1, embed_count)
        self.embed_positions = nn.Embedding(image_token_count, embed_count)
        self.layers: List[DecoderLayerTorch] = nn.ModuleList([
            DecoderLayerTorch(
                image_token_count,
                attention_head_count,
                embed_count,
                glu_embed_count
            ) 
            for _ in range(layer_count)
        ])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        self.lm_head = nn.Linear(embed_count, image_vocab_size + 1, bias=False)
        self.keys_values_state_shape = (
            layer_count * 2 * batch_count,
            image_token_count,
            attention_head_count,
            embed_count // attention_head_count
        )
        self.zero_prob = torch.zeros([1])
        self.token_indices = torch.arange(self.sample_token_count)
        self.start_token = torch.tensor([start_token]).to(torch.long)
        if torch.cuda.is_available():
            self.zero_prob = self.zero_prob.cuda()
            self.token_indices = self.token_indices.cuda()
            self.start_token = self.start_token.cuda()


    def decode_step(self,
        text_tokens: LongTensor,
        encoder_state: FloatTensor,
        keys_values_state: FloatTensor,
        prev_token_and_index: LongTensor
    ) -> Tuple[LongTensor, FloatTensor]:
        attention_mask = text_tokens.not_equal(1)
        batch_count = encoder_state.shape[0]
        prev_token = torch.cat([prev_token_and_index[:1]] * batch_count)
        token_index = torch.cat([prev_token_and_index[1:]] * batch_count)
        decoder_state = self.embed_tokens.forward(prev_token)
        decoder_state += self.embed_positions.forward(token_index)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        decoder_state = decoder_state[:, None]
        keys_values = []
        for i, layer in enumerate(self.layers):
            j1, j2 = i * 2 * batch_count, (i + 1) * 2 * batch_count
            decoder_state, keys_values_layer = layer.forward(
                decoder_state,
                encoder_state,
                keys_values_state[j1:j2],
                attention_mask,
                token_index[:1]
            )
            keys_values.append(keys_values_layer)
        keys_values = torch.cat(keys_values, dim=0)
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        a = self.condition_factor
        logits: FloatTensor = a * logits[0, -1] + (1 - a) * logits[1, -1]

        top_logits = logits.sort(descending=True)[0][:50]
        probs = torch.where(
            logits < top_logits[-1],
            self.zero_prob,
            torch.exp(logits - top_logits[0])
        )
        return probs, keys_values


    def forward(self,
        text_tokens: LongTensor,
        encoder_state: FloatTensor
    ) -> LongTensor:
        image_tokens: List[LongTensor] = []
        keys_values_state = torch.zeros(self.keys_values_state_shape)
        if torch.cuda.is_available(): 
            keys_values_state = keys_values_state.cuda()
        image_token = self.start_token

        for i in range(self.sample_token_count):
            token_index = self.token_indices[i:i+1]
            probs, keys_values_state = self.decode_step(
                text_tokens = text_tokens,
                encoder_state = encoder_state,
                keys_values_state = keys_values_state,
                prev_token_and_index = torch.cat([image_token, token_index])
            )

            image_token = torch.multinomial(probs, 1)
            image_tokens += [image_token]
            
        return torch.cat(image_tokens)