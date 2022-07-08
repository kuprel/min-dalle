from json import load as json_load
from math import sqrt
from os import cpu_count, makedirs
from pathlib import Path
from typing import Iterator

import requests
import torch
from numpy import int32 as np_int32
from numpy import ones as np_ones
from PIL import Image
from torch import FloatTensor, LongTensor

torch.set_grad_enabled(False)
torch.set_num_threads(cpu_count())

from .models import DalleBartDecoder, DalleBartEncoder, VQGanDetokenizer
from .text_tokenizer import TextTokenizer

MIN_DALLE_REPO = "https://huggingface.co/kuprel/min-dalle/resolve/main/"


class MinDalle:
    def __init__(
        self,
        models_root: str = "pretrained",
        dtype: torch.dtype = torch.float32,
        is_mega: bool = True,
        is_reusable: bool = True,
        is_verbose=True,
    ):
        _models_root = Path(models_root)

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

        model_name = f"dalle_bart_{'mega' if is_mega else 'mini'}"
        dalle_path = _models_root / model_name
        vqgan_path = _models_root / "vqgan"

        for _path in [dalle_path, vqgan_path]:
            if not Path(_path).exists():
                makedirs(_path)

        self.vocab_path = dalle_path / "vocab.json"
        self.merges_path = dalle_path / "merges.txt"
        self.encoder_params_path = dalle_path / "encoder.pt"
        self.decoder_params_path = dalle_path / "decoder.pt"
        self.detoker_params_path = vqgan_path / "detoker.pt"

        self.init_tokenizer()
        if is_reusable:
            for _initializer in [
                self.init_encoder,
                self.init_decoder,
                self.init_detokenizer,
            ]:
                _initializer()

    def _verbose_print(self, string: str):
        if self.is_verbose:
            print(string)

    def download_tokenizer(self):
        self._verbose_print("downloading tokenizer params")
        suffix = "" if self.is_mega else "_mini"
        vocab = requests.get(f"{MIN_DALLE_REPO}vocab{suffix}.json")
        merges = requests.get(f"{MIN_DALLE_REPO}merges{suffix}.txt")

        for _path, _data in [(self.vocab_path, vocab), (self.merges_path, merges)]:
            with open(_path, "wb") as f:
                f.write(_data.content)

    def download_encoder(self):
        self._verbose_print("downloading encoder params")
        suffix = "" if self.is_mega else "_mini"
        params = requests.get(f"{MIN_DALLE_REPO}encoder{suffix}.pt")
        with open(self.encoder_params_path, "wb") as f:
            f.write(params.content)

    def download_decoder(self):
        self._verbose_print("downloading decoder params")
        suffix = "" if self.is_mega else "_mini"
        params = requests.get(f"{MIN_DALLE_REPO}decoder{suffix}.pt")
        with open(self.decoder_params_path, "wb") as f:
            f.write(params.content)

    def download_detokenizer(self):
        self._verbose_print("downloading detokenizer params")
        params = requests.get(f"{MIN_DALLE_REPO}detoker.pt")
        with open(self.detoker_params_path, "wb") as f:
            f.write(params.content)

    def init_tokenizer(self):
        is_downloaded = self.vocab_path.exists()
        is_downloaded &= self.merges_path.exists()
        if not is_downloaded:
            self.download_tokenizer()
        self._verbose_print("intializing TextTokenizer")

        with open(self.vocab_path, "r", encoding="utf8") as f:
            vocab = json_load(f)
        with open(self.merges_path, "r", encoding="utf8") as f:
            merges = f.read().split("\n")[1:-1]

        self.tokenizer = TextTokenizer(vocab, merges)

    def init_encoder(self):
        is_downloaded = self.encoder_params_path.exists()
        if not is_downloaded:
            self.download_encoder()
        self._verbose_print("initializing DalleBartEncoder")
        self.encoder = (
            DalleBartEncoder(
                attention_head_count=self.attention_head_count,
                embed_count=self.embed_count,
                glu_embed_count=self.glu_embed_count,
                text_token_count=self.text_token_count,
                text_vocab_count=self.text_vocab_count,
                layer_count=self.layer_count,
            )
            .to(self.dtype)
            .eval()
        )
        params = torch.load(self.encoder_params_path)
        self.encoder.load_state_dict(params, strict=False)
        del params
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()

    def init_decoder(self):
        is_downloaded = self.decoder_params_path.exists()
        if not is_downloaded:
            self.download_decoder()
        self._verbose_print("initializing DalleBartDecoder")
        self.decoder = (
            DalleBartDecoder(
                image_vocab_count=self.image_vocab_count,
                attention_head_count=self.attention_head_count,
                embed_count=self.embed_count,
                glu_embed_count=self.glu_embed_count,
                layer_count=self.layer_count,
                start_token=self.image_vocab_count,
            )
            .to(self.dtype)
            .eval()
        )
        params = torch.load(self.decoder_params_path)
        self.decoder.load_state_dict(params, strict=False)
        del params
        if torch.cuda.is_available():
            self.decoder = self.decoder.cuda()

    def init_detokenizer(self):
        is_downloaded = self.detoker_params_path.exists()
        if not is_downloaded:
            self.download_detokenizer()
        self._verbose_print("initializing VQGanDetokenizer")
        self.detokenizer = VQGanDetokenizer().eval()
        params = torch.load(self.detoker_params_path)
        self.detokenizer.load_state_dict(params)
        del params
        if torch.cuda.is_available():
            self.detokenizer = self.detokenizer.cuda()

    def images_from_tokens(
        self, image_tokens: LongTensor, is_verbose: bool = False
    ) -> FloatTensor:
        if not self.is_reusable:
            del self.decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not self.is_reusable:
            self.init_detokenizer()
        if is_verbose:
            print("detokenizing image")
        images = self.detokenizer.forward(image_tokens).to(torch.uint8)
        if not self.is_reusable:
            del self.detokenizer
        return images

    def grid_from_images(self, images: FloatTensor) -> Image.Image:
        grid_size = int(sqrt(images.shape[0]))
        images = images.reshape([grid_size] * 2 + list(images.shape[1:]))
        image = images.flatten(1, 2).transpose(0, 1).flatten(1, 2)
        image = Image.fromarray(image.to("cpu").detach().numpy())
        return image

    def generate_images_stream(
        self,
        text: str,
        seed: int,
        image_count: int,
        log2_mid_count: int,
        log2_k: int = 6,
        log2_supercondition_factor: int = 3,
        is_verbose: bool = False,
    ) -> Iterator[FloatTensor]:
        assert log2_mid_count in range(5)
        if is_verbose:
            print("tokenizing text")
        tokens = self.tokenizer.tokenize(text, is_verbose=is_verbose)
        if len(tokens) > self.text_token_count:
            tokens = tokens[: self.text_token_count]
        if is_verbose:
            print("text tokens", tokens)
        text_tokens = np_ones((2, 64), dtype=np_int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, : len(tokens)] = tokens

        text_tokens = torch.tensor(text_tokens).to(torch.long)
        if torch.cuda.is_available():
            text_tokens = text_tokens.cuda()

        if not self.is_reusable:
            self.init_encoder()
        if is_verbose:
            print("encoding text tokens")
        with torch.cuda.amp.autocast(dtype=self.dtype):
            encoder_state = self.encoder.forward(text_tokens)
        if not self.is_reusable:
            del self.encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not self.is_reusable:
            self.init_decoder()

        with torch.cuda.amp.autocast(dtype=self.dtype):
            (
                encoder_state,
                attention_mask,
                attention_state,
                image_tokens,
            ) = self.decoder.decode_initial(
                seed, image_count, text_tokens, encoder_state
            )

        row_count = 16
        for row_index in range(row_count):
            if is_verbose:
                print(f"sampling row {row_index + 1} of {row_count}")
            with torch.cuda.amp.autocast(dtype=self.dtype):
                attention_state, image_tokens = self.decoder.decode_row(
                    row_index,
                    log2_k,
                    log2_supercondition_factor,
                    encoder_state,
                    attention_mask,
                    attention_state,
                    image_tokens,
                )
            with torch.cuda.amp.autocast(dtype=torch.float32):
                if ((row_index + 1) * (2**log2_mid_count)) % row_count == 0:
                    tokens = image_tokens[:, 1:]
                    yield self.images_from_tokens(tokens, is_verbose)

    def generate_image_stream(
        self,
        text: str,
        seed: int,
        grid_size: int,
        log2_mid_count: int,
        log2_k: int = 6,
        log2_supercondition_factor: int = 3,
        is_verbose: bool = False,
    ) -> Iterator[Image.Image]:
        images_stream = self.generate_images_stream(
            text,
            seed,
            grid_size**2,
            log2_mid_count,
            log2_k,
            log2_supercondition_factor,
            is_verbose,
        )
        for images in images_stream:
            yield self.grid_from_images(images)

    def generate_images(
        self,
        text: str,
        seed: int = -1,
        image_count: int = 1,
        log2_k: int = 6,
        log2_supercondition_factor: int = 3,
        is_verbose: bool = False,
    ) -> FloatTensor:
        log2_mid_count = 0
        images_stream = self.generate_images_stream(
            text,
            seed,
            image_count,
            log2_mid_count,
            log2_k,
            log2_supercondition_factor,
            is_verbose,
        )
        return next(images_stream)

    def generate_image(
        self,
        text: str,
        seed: int = -1,
        grid_size: int = 1,
        log2_k: int = 6,
        log2_supercondition_factor: int = 3,
        is_verbose: bool = False,
    ) -> Image.Image:
        log2_mid_count = 0
        image_stream = self.generate_image_stream(
            text,
            seed,
            grid_size,
            log2_mid_count,
            log2_k,
            log2_supercondition_factor,
            is_verbose,
        )
        return next(image_stream)
