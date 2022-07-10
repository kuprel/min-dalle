import torch
from torch import FloatTensor, LongTensor
from torch.nn import Module, ModuleList, GroupNorm, Conv2d, Embedding


class ResnetBlock(Module):
    def __init__(self, log2_count_in: int, log2_count_out: int):
        super().__init__()
        m, n = 2 ** log2_count_in, 2 ** log2_count_out
        self.is_middle = m == n
        self.norm1 = GroupNorm(2 ** 5, m)
        self.conv1 = Conv2d(m, n, 3, padding=1)
        self.norm2 = GroupNorm(2 ** 5, n)
        self.conv2 = Conv2d(n, n, 3, padding=1)
        if not self.is_middle:
            self.nin_shortcut = Conv2d(m, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        h = x
        h = self.norm1.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv1.forward(h)
        h = self.norm2.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv2(h)
        if not self.is_middle:
            x = self.nin_shortcut.forward(x)
        return x + h


class AttentionBlock(Module):
    def __init__(self):
        super().__init__()
        n = 2 ** 9
        self.norm = GroupNorm(2 ** 5, n)
        self.q = Conv2d(n, n, 1)
        self.k = Conv2d(n, n, 1)
        self.v = Conv2d(n, n, 1)
        self.proj_out = Conv2d(n, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        n, m = 2 ** 9, x.shape[0]
        h = x
        h = self.norm(h)
        q = self.q.forward(h)
        k = self.k.forward(h)
        v = self.v.forward(h)
        q = q.reshape(m, n, 2 ** 8)
        q = q.permute(0, 2, 1)
        k = k.reshape(m, n, 2 ** 8)
        w = torch.bmm(q, k)
        w /= n ** 0.5
        w = torch.softmax(w, dim=2)
        v = v.reshape(m, n, 2 ** 8)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        h = h.reshape(m, n, 2 ** 4, 2 ** 4)
        h = self.proj_out.forward(h)
        return x + h


class MiddleLayer(Module):
    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)
    
    def forward(self, h: FloatTensor) -> FloatTensor:
        h = self.block_1.forward(h)
        h = self.attn_1.forward(h)
        h = self.block_2.forward(h)
        return h


class Upsample(Module):
    def __init__(self, log2_count):
        super().__init__()
        n = 2 ** log2_count
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = Conv2d(n, n, 3, padding=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.upsample.forward(x.to(torch.float32))
        x = self.conv.forward(x)
        return x


class UpsampleBlock(Module):
    def __init__(
        self, 
        log2_count_in: int, 
        log2_count_out: int, 
        has_attention: bool, 
        has_upsample: bool
    ):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample
        self.block = ModuleList([
            ResnetBlock(log2_count_in, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out)
        ])
        if has_attention:
            self.attn = ModuleList([
                AttentionBlock(),
                AttentionBlock(),
                AttentionBlock()
            ])
        else:
            self.attn = ModuleList()

        if has_upsample:
            self.upsample = Upsample(log2_count_out)


    def forward(self, h: FloatTensor) -> FloatTensor:
        for j in range(3):
            h = self.block[j].forward(h)
            if self.has_attention:
                h = self.attn[j].forward(h)
        if self.has_upsample:
            h = self.upsample.forward(h)
        return h


class Decoder(Module):
    def __init__(self):
        super().__init__()

        self.conv_in = Conv2d(2 ** 8, 2 ** 9, 3, padding=1)
        self.mid = MiddleLayer()

        self.up = ModuleList([
            UpsampleBlock(7, 7, False, False),
            UpsampleBlock(8, 7, False, True),
            UpsampleBlock(8, 8, False, True),
            UpsampleBlock(9, 8, False, True),
            UpsampleBlock(9, 9, True, True)
        ])

        self.norm_out = GroupNorm(2 ** 5, 2 ** 7)
        self.conv_out = Conv2d(2 ** 7, 3, 3, padding=1)

    def forward(self, z: FloatTensor) -> FloatTensor:
        z = self.conv_in.forward(z)
        z = self.mid.forward(z)

        for i in reversed(range(5)):
            z = self.up[i].forward(z)

        z = self.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z)
        return z


class VQGanDetokenizer(Module):
    def __init__(self):
        super().__init__()
        vocab_count, embed_count = 2 ** 14, 2 ** 8
        self.vocab_count = vocab_count
        self.embedding = Embedding(vocab_count, embed_count)
        self.post_quant_conv = Conv2d(embed_count, embed_count, 1)
        self.decoder = Decoder()

    def forward(self, z: LongTensor) -> FloatTensor:
        z.clamp_(0, self.vocab_count - 1)
        z = self.embedding.forward(z)
        z = z.view((z.shape[0], 2 ** 4, 2 ** 4, 2 ** 8))
        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv.forward(z)
        z = self.decoder.forward(z)
        z = z.permute(0, 2, 3, 1)
        z = z.clip(0.0, 1.0) * 255
        return z
