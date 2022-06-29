import jax
import numpy
from PIL import Image
import torch

from .min_dalle import MinDalle
from .models.dalle_bart_encoder_flax import DalleBartEncoderFlax
from .models.dalle_bart_decoder_flax import DalleBartDecoderFlax


class MinDalleFlax(MinDalle):
    def __init__(self, is_mega: bool):
        super().__init__(is_mega)
        print("initializing MinDalleFlax")

        print("loading encoder")
        self.encoder = DalleBartEncoderFlax(
            attention_head_count = self.config['encoder_attention_heads'],
            embed_count = self.config['d_model'],
            glu_embed_count = self.config['encoder_ffn_dim'],
            text_token_count = self.config['max_text_length'],
            text_vocab_count = self.config['encoder_vocab_size'],
            layer_count = self.config['encoder_layers']
        ).bind({'params': self.model_params.pop('encoder')})

        print("loading decoder")
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
        

    def generate_image(self, text: str, seed: int) -> Image.Image:
        text_tokens = self.tokenize_text(text)

        print("encoding text tokens")
        encoder_state = self.encoder(text_tokens)

        print("sampling image tokens")
        image_tokens = self.decoder.sample_image_tokens(
            text_tokens,
            encoder_state,
            jax.random.PRNGKey(seed),
            self.model_params['decoder']
        )

        image_tokens = torch.tensor(numpy.array(image_tokens))

        print("detokenizing image")
        image = self.detokenizer.forward(image_tokens).to(torch.uint8)
        image = Image.fromarray(image.to('cpu').detach().numpy())
        return image