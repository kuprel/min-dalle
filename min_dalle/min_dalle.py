import os
from PIL import Image
import numpy
from torch import LongTensor
import torch
import json
import requests
from typing import Iterator
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())

from .text_tokenizer import TextTokenizer
from .models import DalleBartEncoder, DalleBartDecoder, VQGanDetokenizer

MIN_DALLE_REPO = 'https://huggingface.co/kuprel/min-dalle/resolve/main/'


class MinDalle:
    def __init__(
        self,
        models_root: str = 'pretrained',
        dtype: torch.dtype = torch.float32,
        is_mega: bool = True, 
        is_reusable: bool = True,
        is_verbose = True
    ):
        self.is_mega = is_mega
        self.is_reusable = is_reusable
        self.dtype = dtype
        self.is_verbose = is_verbose
        self.text_token_count = 64
        self.layer_count = 24 if is_mega else 12
        self.attention_head_count = 32 if is_mega else 16
        self.embed_count = 2048 if is_mega else 1024
        self.glu_embed_count = 4096 if is_mega else 2730
        self.text_vocab_count = 50272 if is_mega else 50264
        self.image_vocab_count = 16415 if is_mega else 16384

        model_name = 'dalle_bart_{}'.format('mega' if is_mega else 'mini')
        dalle_path = os.path.join(models_root, model_name)
        vqgan_path = os.path.join(models_root, 'vqgan')
        if not os.path.exists(dalle_path): os.makedirs(dalle_path)
        if not os.path.exists(vqgan_path): os.makedirs(vqgan_path)
        self.vocab_path = os.path.join(dalle_path, 'vocab.json')
        self.merges_path = os.path.join(dalle_path, 'merges.txt')
        self.encoder_params_path = os.path.join(dalle_path, 'encoder.pt')
        self.decoder_params_path = os.path.join(dalle_path, 'decoder.pt')
        self.detoker_params_path = os.path.join(vqgan_path, 'detoker.pt')

        self.init_tokenizer()
        if is_reusable:
            self.init_encoder()
            self.init_decoder()
            self.init_detokenizer()


    def download_tokenizer(self):
        if self.is_verbose: print("downloading tokenizer params")
        suffix = '' if self.is_mega else '_mini'
        vocab = requests.get(MIN_DALLE_REPO + 'vocab{}.json'.format(suffix))
        merges = requests.get(MIN_DALLE_REPO + 'merges{}.txt'.format(suffix))
        with open(self.vocab_path, 'wb') as f: f.write(vocab.content)
        with open(self.merges_path, 'wb') as f: f.write(merges.content)


    def download_encoder(self):
        if self.is_verbose: print("downloading encoder params")
        suffix = '' if self.is_mega else '_mini'
        params = requests.get(MIN_DALLE_REPO + 'encoder{}.pt'.format(suffix))
        with open(self.encoder_params_path, 'wb') as f: f.write(params.content)


    def download_decoder(self):
        if self.is_verbose: print("downloading decoder params")
        suffix = '' if self.is_mega else '_mini'
        params = requests.get(MIN_DALLE_REPO + 'decoder{}.pt'.format(suffix))
        with open(self.decoder_params_path, 'wb') as f: f.write(params.content)
    

    def download_detokenizer(self):
        if self.is_verbose: print("downloading detokenizer params")
        params = requests.get(MIN_DALLE_REPO + 'detoker.pt')
        with open(self.detoker_params_path, 'wb') as f: f.write(params.content)


    def init_tokenizer(self):
        is_downloaded = os.path.exists(self.vocab_path)
        is_downloaded &= os.path.exists(self.merges_path)
        if not is_downloaded: self.download_tokenizer()
        if self.is_verbose: print("intializing TextTokenizer")
        with open(self.vocab_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        with open(self.merges_path, 'r', encoding='utf8') as f:
            merges = f.read().split("\n")[1:-1]
        self.tokenizer = TextTokenizer(vocab, merges)


    def init_encoder(self):
        is_downloaded = os.path.exists(self.encoder_params_path)
        if not is_downloaded: self.download_encoder()
        if self.is_verbose: print("initializing DalleBartEncoder")
        self.encoder = DalleBartEncoder(
            attention_head_count = self.attention_head_count,
            embed_count = self.embed_count,
            glu_embed_count = self.glu_embed_count,
            text_token_count = self.text_token_count,
            text_vocab_count = self.text_vocab_count,
            layer_count = self.layer_count
        ).to(self.dtype).eval()
        params = torch.load(self.encoder_params_path)
        self.encoder.load_state_dict(params, strict=False)
        del params
        if torch.cuda.is_available(): self.encoder = self.encoder.cuda()


    def init_decoder(self):
        is_downloaded = os.path.exists(self.decoder_params_path)
        if not is_downloaded: self.download_decoder()
        if self.is_verbose: print("initializing DalleBartDecoder")
        self.decoder = DalleBartDecoder(
            image_vocab_count = self.image_vocab_count,
            attention_head_count = self.attention_head_count,
            embed_count = self.embed_count,
            glu_embed_count = self.glu_embed_count,
            layer_count = self.layer_count,
            start_token = self.image_vocab_count
        ).to(self.dtype).eval()
        params = torch.load(self.decoder_params_path)
        self.decoder.load_state_dict(params, strict=False)
        del params
        if torch.cuda.is_available(): self.decoder = self.decoder.cuda()


    def init_detokenizer(self):
        is_downloaded = os.path.exists(self.detoker_params_path)
        if not is_downloaded: self.download_detokenizer()
        if self.is_verbose: print("initializing VQGanDetokenizer")
        self.detokenizer = VQGanDetokenizer().eval()
        params = torch.load(self.detoker_params_path)
        self.detokenizer.load_state_dict(params)
        del params
        if torch.cuda.is_available(): self.detokenizer = self.detokenizer.cuda()


    def image_from_tokens(
        self,
        grid_size: int,
        image_tokens: LongTensor,
        is_verbose: bool = False
    ) -> Image.Image:
        if not self.is_reusable: del self.decoder
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if not self.is_reusable: self.init_detokenizer()
        if is_verbose: print("detokenizing image")
        images = self.detokenizer.forward(image_tokens).to(torch.uint8)
        if not self.is_reusable: del self.detokenizer
        images = images.reshape([grid_size] * 2 + list(images.shape[1:]))
        image = images.flatten(1, 2).transpose(0, 1).flatten(1, 2)
        image = Image.fromarray(image.to('cpu').detach().numpy())
        return image


    def generate_image_stream(
        self, 
        text: str, 
        seed: int,
        grid_size: int,
        log2_mid_count: int,
        log2_k: int = 6,
        log2_supercondition_factor: int = 3,
        is_verbose: bool = False
    ) -> Iterator[Image.Image]:
        assert(log2_mid_count in range(5))
        if is_verbose: print("tokenizing text")
        tokens = self.tokenizer.tokenize(text, is_verbose=is_verbose)
        if len(tokens) > self.text_token_count: 
            tokens = tokens[:self.text_token_count]
        if is_verbose: print("text tokens", tokens)
        text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, :len(tokens)] = tokens

        text_tokens = torch.tensor(text_tokens).to(torch.long)
        if torch.cuda.is_available(): text_tokens = text_tokens.cuda()

        if not self.is_reusable: self.init_encoder()
        if is_verbose: print("encoding text tokens")
        with torch.cuda.amp.autocast(dtype=self.dtype):
            encoder_state = self.encoder.forward(text_tokens)
        if not self.is_reusable: del self.encoder
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if not self.is_reusable: self.init_decoder()

        with torch.cuda.amp.autocast(dtype=self.dtype):
            encoder_state, attention_mask, attention_state, image_tokens = ( 
                self.decoder.decode_initial(
                    seed, 
                    grid_size ** 2, 
                    text_tokens, 
                    encoder_state
                )
            )

        row_count = 16
        for row_index in range(row_count):
            if is_verbose: 
                print('sampling row {} of {}'.format(row_index + 1, row_count))
            with torch.cuda.amp.autocast(dtype=self.dtype):
                attention_state, image_tokens = self.decoder.decode_row(
                    row_index,
                    log2_k,
                    log2_supercondition_factor,
                    encoder_state,
                    attention_mask,
                    attention_state,
                    image_tokens
                )
            with torch.cuda.amp.autocast(dtype=torch.float32):
                if ((row_index + 1) * (2 ** log2_mid_count)) % row_count == 0:
                    tokens = image_tokens[:, 1:]
                    image = self.image_from_tokens(grid_size, tokens, is_verbose)
                    yield image


    def generate_image(
        self, 
        text: str,
        seed: int = -1,
        grid_size: int = 1,
        log2_k: int = 6,
        log2_supercondition_factor: int = 3,
        is_verbose: bool = False
    ) -> Image.Image:
        log2_mid_count = 0
        image_stream = self.generate_image_stream(
            text,
            seed,
            grid_size,
            log2_mid_count,
            log2_k,
            log2_supercondition_factor,
            is_verbose
        )
        return next(image_stream)