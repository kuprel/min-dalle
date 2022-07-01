import os
from PIL import Image
import numpy
from torch import LongTensor
import torch
import json
import requests
import random
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())

from .text_tokenizer import TextTokenizer
from .models import DalleBartEncoder, DalleBartDecoder, VQGanDetokenizer

MIN_DALLE_REPO = 'https://huggingface.co/kuprel/min-dalle/resolve/main/'


class MinDalle:
    def __init__(
        self,
        is_mega: bool, 
        is_reusable: bool = True,
        models_root: str = 'pretrained',
        sample_token_count: int = 256
    ):
        print("initializing MinDalleTorch")
        self.is_mega = is_mega
        self.is_reusable = is_reusable
        self.sample_token_count = sample_token_count
        self.batch_count = 2
        self.text_token_count = 64
        self.image_token_count = 256
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
        print("downloading tokenizer params")
        suffix = '' if self.is_mega else '_mini'
        vocab = requests.get(MIN_DALLE_REPO + 'vocab{}.json'.format(suffix))
        merges = requests.get(MIN_DALLE_REPO + 'merges{}.txt'.format(suffix))
        with open(self.vocab_path, 'wb') as f: f.write(vocab.content)
        with open(self.merges_path, 'wb') as f: f.write(merges.content)


    def download_encoder(self):
        print("downloading encoder params")
        suffix = '' if self.is_mega else '_mini'
        params = requests.get(MIN_DALLE_REPO + 'encoder{}.pt'.format(suffix))
        with open(self.encoder_params_path, 'wb') as f: f.write(params.content)


    def download_decoder(self):
        print("downloading decoder params")
        suffix = '' if self.is_mega else '_mini'
        params = requests.get(MIN_DALLE_REPO + 'decoder{}.pt'.format(suffix))
        with open(self.decoder_params_path, 'wb') as f: f.write(params.content)
    

    def download_detokenizer(self):
        print("downloading detokenizer params")
        params = requests.get(MIN_DALLE_REPO + 'detoker.pt')
        with open(self.detoker_params_path, 'wb') as f: f.write(params.content)


    def init_tokenizer(self):
        is_downloaded = os.path.exists(self.vocab_path)
        is_downloaded &= os.path.exists(self.merges_path)
        if not is_downloaded: self.download_tokenizer()
        print("intializing TextTokenizer")
        with open(self.vocab_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        with open(self.merges_path, 'r', encoding='utf8') as f:
            merges = f.read().split("\n")[1:-1]
        self.tokenizer = TextTokenizer(vocab, merges)


    def init_encoder(self):
        is_downloaded = os.path.exists(self.encoder_params_path)
        if not is_downloaded: self.download_encoder()
        print("initializing DalleBartEncoderTorch")
        self.encoder = DalleBartEncoder(
            attention_head_count = self.attention_head_count,
            embed_count = self.embed_count,
            glu_embed_count = self.glu_embed_count,
            text_token_count = self.text_token_count,
            text_vocab_count = self.text_vocab_count,
            layer_count = self.layer_count
        )
        params = torch.load(self.encoder_params_path)
        self.encoder.load_state_dict(params, strict=False)
        del params
        if torch.cuda.is_available(): self.encoder = self.encoder.cuda()


    def init_decoder(self):
        is_downloaded = os.path.exists(self.decoder_params_path)
        if not is_downloaded: self.download_decoder()
        print("initializing DalleBartDecoderTorch")
        self.decoder = DalleBartDecoder(
            sample_token_count = self.sample_token_count,
            image_token_count = self.image_token_count,
            image_vocab_count = self.image_vocab_count,
            attention_head_count = self.attention_head_count,
            embed_count = self.embed_count,
            glu_embed_count = self.glu_embed_count,
            layer_count = self.layer_count,
            start_token = self.image_vocab_count,
            batch_count = self.batch_count
        )
        params = torch.load(self.decoder_params_path)
        self.decoder.load_state_dict(params, strict=False)
        del params
        if torch.cuda.is_available(): self.decoder = self.decoder.cuda()


    def init_detokenizer(self):
        is_downloaded = os.path.exists(self.detoker_params_path)
        if not is_downloaded: self.download_detokenizer()
        print("initializing VQGanDetokenizer")
        self.detokenizer = VQGanDetokenizer()
        params = torch.load(self.detoker_params_path)
        self.detokenizer.load_state_dict(params)
        del params
        if torch.cuda.is_available(): self.detokenizer = self.detokenizer.cuda()


    def generate_image_tokens(self, text: str, seed: int) -> LongTensor:
        print("tokenizing text")
        tokens = self.tokenizer.tokenize(text)
        print("text tokens", tokens)
        text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, :len(tokens)] = tokens

        text_tokens = torch.tensor(text_tokens).to(torch.long)
        if torch.cuda.is_available(): text_tokens = text_tokens.cuda()

        if not self.is_reusable: self.init_encoder()
        print("encoding text tokens")
        encoder_state = self.encoder.forward(text_tokens)
        if not self.is_reusable: del self.encoder

        if not self.is_reusable: self.init_decoder()
        print("sampling image tokens")
        if seed < 0: seed = random.randint(0, 2 ** 31)
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