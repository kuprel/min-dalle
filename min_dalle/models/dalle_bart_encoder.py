from typing import List
import torch
from torch import nn, BoolTensor, FloatTensor, LongTensor


class GLU(nn.Module):
    def __init__(self, count_in_out: int, count_middle: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.ln0 = nn.LayerNorm(count_in_out)
        self.ln1 = nn.LayerNorm(count_middle)
        self.fc0 = nn.Linear(count_in_out, count_middle, bias=False)
        self.fc1 = nn.Linear(count_in_out, count_middle, bias=False)
        self.fc2 = nn.Linear(count_middle, count_in_out, bias=False)
    
    def forward(self, z: FloatTensor) -> FloatTensor:
        z = self.ln0.forward(z)
        w = self.fc0.forward(z)
        w = self.gelu.forward(w)
        v = self.fc1.forward(z)
        z = self.ln1.forward(w * v)
        z = self.fc2.forward(z)
        return z


class AttentionBase(nn.Module):
    def __init__(self, head_count: int, embed_count: int):
        super().__init__()
        self.head_count = head_count
        self.embed_count = embed_count

        self.k_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.v_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.q_proj = nn.Linear(embed_count, embed_count, bias=False)
        self.out_proj = nn.Linear(embed_count, embed_count, bias=False)
    
    def forward(
        self,
        keys: FloatTensor,
        values: FloatTensor,
        queries: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = keys.reshape(keys.shape[:2] + (self.head_count, -1))
        values = values.reshape(values.shape[:2] + (self.head_count, -1))
        queries = queries.reshape(queries.shape[:2] + (self.head_count, -1))
        queries /= queries.shape[-1] ** 0.5

        attention_bias = (1 - attention_mask.to(torch.float32)) * -1e12
        attention_weights: FloatTensor = torch.einsum(
            'bqhc,bkhc->bhqk',
            queries, 
            keys
        )
        attention_weights += attention_bias[:, None, None, :]
        attention_weights = torch.softmax(attention_weights, -1)
        attention_output: FloatTensor = torch.einsum(
            "bhqk,bkhc->bqhc",
            attention_weights, 
            values
        )
        shape = attention_output.shape[:2] + (self.embed_count,)
        attention_output = attention_output.reshape(shape)
        attention_output = self.out_proj.forward(attention_output)
        return attention_output


class EncoderSelfAttention(AttentionBase):
    def forward(
        self,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(encoder_state)
        return super().forward(keys, values, queries, attention_mask)


class EncoderLayer(nn.Module):
    def __init__(self, embed_count: int, head_count: int, glu_embed_count: int):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = EncoderSelfAttention(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLU(embed_count, glu_embed_count)
    
    def forward(
        self,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        residual = encoder_state
        encoder_state = self.pre_self_attn_layer_norm.forward(encoder_state)
        encoder_state = self.self_attn.forward(encoder_state, attention_mask)
        encoder_state = self.self_attn_layer_norm.forward(encoder_state)
        encoder_state = residual + encoder_state
        residual = encoder_state
        encoder_state = self.glu.forward(encoder_state)
        encoder_state = residual + encoder_state
        return encoder_state


class DalleBartEncoder(nn.Module):
    def __init__(
        self,
        layer_count: int,
        embed_count: int,
        attention_head_count: int,
        text_vocab_count: int,
        text_token_count: int,
        glu_embed_count: int,
        device: str
    ):
        super().__init__()
        self.text_vocab_count = text_vocab_count
        self.embed_tokens = nn.Embedding(text_vocab_count, embed_count)
        self.embed_positions = nn.Embedding(text_token_count, embed_count)
        self.layers: List[EncoderLayer] = nn.ModuleList([
            EncoderLayer(
                embed_count = embed_count,
                head_count = attention_head_count,
                glu_embed_count = glu_embed_count
            ) 
            for _ in range(layer_count)
        ])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        token_indices = torch.arange(text_token_count, device=device)
        self.pose_tokens = torch.stack([token_indices] * 2)

    def forward(self, text_tokens: LongTensor) -> FloatTensor:
        attention_mask = text_tokens.not_equal(1)
        encoder_state = (
            self.embed_tokens.forward(text_tokens) +
            self.embed_positions.forward(self.pose_tokens)
        )
        encoder_state = self.layernorm_embedding.forward(encoder_state)
        for layer in self.layers:
            encoder_state = layer.forward(encoder_state, attention_mask)
        encoder_state = self.final_ln.forward(encoder_state)
        return encoder_state