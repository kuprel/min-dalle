import os
from PIL import Image
from typing import Dict
from torch import LongTensor
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())

from .min_dalle_base import MinDalleBase
from .models.dalle_bart_encoder_torch import DalleBartEncoderTorch
from .models.dalle_bart_decoder_torch import DalleBartDecoderTorch
from .models.vqgan_detokenizer import VQGanDetokenizer


class MinDalleTorch(MinDalleBase):
    def __init__(
        self, 
        is_mega: bool, 
        is_reusable: bool = True,
        token_count: int = 256
    ):
        print("initializing MinDalleTorch")
        super().__init__(is_mega)
        self.is_reusable = is_reusable
        self.token_count = token_count
    
        self.encoder_params_path = os.path.join(self.model_path, 'encoder.pt')
        self.decoder_params_path = os.path.join(self.model_path, 'decoder.pt')
        self.detoker_params_path = os.path.join('pretrained', 'vqgan', 'detoker.pt')

        if is_reusable:
            self.init_encoder()
            self.init_decoder()
            self.init_detokenizer()


    def init_encoder(self):
        print("initializing DalleBartEncoderTorch")
        self.encoder = DalleBartEncoderTorch(
            attention_head_count = 32 if self.is_mega else 16,
            embed_count = 2048 if self.is_mega else 1024,
            glu_embed_count = 4096 if self.is_mega else 2730,
            text_token_count = 64,
            text_vocab_count = 50272 if self.is_mega else 50264,
            layer_count = 24 if self.is_mega else 12
        )
        params = torch.load(self.encoder_params_path)
        self.encoder.load_state_dict(params, strict=False)
        del params
        if torch.cuda.is_available(): self.encoder = self.encoder.cuda()


    def init_decoder(self):
        print("initializing DalleBartDecoderTorch")
        self.decoder = DalleBartDecoderTorch(
            sample_token_count = self.token_count,
            image_token_count = 256,
            image_vocab_count = 16415 if self.is_mega else 16384,
            attention_head_count = 32 if self.is_mega else 16,
            embed_count = 2048 if self.is_mega else 1024,
            glu_embed_count = 4096 if self.is_mega else 2730,
            layer_count = 24 if self.is_mega else 12,
            start_token = 16415 if self.is_mega else 16384,
            batch_count = 2
        )
        params = torch.load(self.decoder_params_path)
        self.decoder.load_state_dict(params, strict=False)
        del params
        if torch.cuda.is_available(): self.decoder = self.decoder.cuda()


    def init_detokenizer(self):
        print("initializing VQGanDetokenizer")
        self.detokenizer = VQGanDetokenizer()
        params = torch.load(self.detoker_params_path)
        self.detokenizer.load_state_dict(params)
        del params
        if torch.cuda.is_available(): self.detokenizer = self.detokenizer.cuda()
            

    def generate_image_tokens(self, text: str, seed: int) -> LongTensor:
        text_tokens = self.tokenize_text(text)
        text_tokens = torch.tensor(text_tokens).to(torch.long)
        if torch.cuda.is_available(): text_tokens = text_tokens.cuda()

        if not self.is_reusable: self.init_encoder()
        print("encoding text tokens")
        encoder_state = self.encoder.forward(text_tokens)
        if not self.is_reusable: del self.encoder

        if not self.is_reusable: self.init_decoder()
        print("sampling image tokens")
        torch.manual_seed(seed)
        image_tokens = self.decoder.forward(text_tokens, encoder_state)
        if not self.is_reusable: del self.decoder
        return image_tokens
        

    def generate_image(self, text: str, seed: int) -> Image.Image:
        image_tokens = self.generate_image_tokens(text, seed)
        if not self.is_reusable: self.init_detokenizer()
        print("detokenizing image")
        image = self.detokenizer.forward(image_tokens).to(torch.uint8)
        if not self.is_reusable: del self.detokenizer
        image = Image.fromarray(image.to('cpu').detach().numpy())
        return image