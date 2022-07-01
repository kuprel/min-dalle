import jax
import numpy
from PIL import Image
import torch

from .min_dalle_base import MinDalleBase
from .models.dalle_bart_encoder_flax import DalleBartEncoderFlax
from .models.dalle_bart_decoder_flax import DalleBartDecoderFlax
from .models.vqgan_detokenizer import VQGanDetokenizer

from .load_params import load_dalle_bart_flax_params, load_vqgan_torch_params


class MinDalleFlax(MinDalleBase):
    def __init__(self, is_mega: bool, is_reusable: bool = True):
        super().__init__(is_mega)
        self.is_reusable = is_reusable
        print("initializing MinDalleFlax")
        self.model_params = load_dalle_bart_flax_params(self.model_path)
        if is_reusable:
            self.init_encoder()
            self.init_decoder()
            self.init_detokenizer()


    def init_encoder(self):
        print("initializing DalleBartEncoderFlax")
        self.encoder: DalleBartEncoderFlax = DalleBartEncoderFlax(
            attention_head_count = self.config['encoder_attention_heads'],
            embed_count = self.config['d_model'],
            glu_embed_count = self.config['encoder_ffn_dim'],
            text_token_count = self.config['max_text_length'],
            text_vocab_count = self.config['encoder_vocab_size'],
            layer_count = self.config['encoder_layers']
        ).bind({'params': self.model_params.pop('encoder')})
        

    def init_decoder(self):
        print("initializing DalleBartDecoderFlax")
        self.decoder = DalleBartDecoderFlax(
            image_token_count = self.config['image_length'],
            text_token_count = self.config['max_text_length'],
            image_vocab_count = self.config['image_vocab_size'],
            attention_head_count = self.config['decoder_attention_heads'],
            embed_count = self.config['d_model'],
            glu_embed_count = self.config['decoder_ffn_dim'],
            layer_count = self.config['decoder_layers'],
            start_token = self.config['decoder_start_token_id']
        )


    def init_detokenizer(self):
        print("initializing VQGanDetokenizer")
        params = load_vqgan_torch_params('./pretrained/vqgan')
        self.detokenizer = VQGanDetokenizer()
        self.detokenizer.load_state_dict(params)
        del params

    def generate_image(self, text: str, seed: int) -> Image.Image:
        text_tokens = self.tokenize_text(text)

        if not self.is_reusable: self.init_encoder()
        print("encoding text tokens")
        encoder_state = self.encoder(text_tokens)
        if not self.is_reusable: del self.encoder

        if not self.is_reusable:
            self.init_decoder()
            params = self.model_params.pop('decoder')
        else:
            params = self.model_params['decoder']
        print("sampling image tokens")
        image_tokens = self.decoder.sample_image_tokens(
            text_tokens,
            encoder_state,
            jax.random.PRNGKey(seed),
            params
        )
        if not self.is_reusable: del self.decoder

        image_tokens = torch.tensor(numpy.array(image_tokens))

        if not self.is_reusable: self.init_detokenizer()
        print("detokenizing image")
        image = self.detokenizer.forward(image_tokens).to(torch.uint8)
        if not self.is_reusable: del self.detokenizer
        image = Image.fromarray(image.to('cpu').detach().numpy())
        return image