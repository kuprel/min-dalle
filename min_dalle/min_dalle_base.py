import os
import json
import numpy

from .text_tokenizer import TextTokenizer

class MinDalleBase:
    def __init__(self, is_mega: bool):
        self.is_mega = is_mega
        model_name = 'dalle_bart_{}'.format('mega' if is_mega else 'mini')
        self.model_path = os.path.join('pretrained', model_name)

        print("reading files from {}".format(self.model_path))
        config_path = os.path.join(self.model_path, 'config.json')
        vocab_path = os.path.join(self.model_path, 'vocab.json')
        merges_path = os.path.join(self.model_path, 'merges.txt')

        with open(config_path, 'r', encoding='utf8') as f: 
            self.config = json.load(f)
        with open(vocab_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        with open(merges_path, 'r', encoding='utf8') as f:
            merges = f.read().split("\n")[1:-1]
            
        self.tokenizer = TextTokenizer(vocab, merges)


    def tokenize_text(self, text: str) -> numpy.ndarray:
        print("tokenizing text")
        tokens = self.tokenizer.tokenize(text)
        print("text tokens", tokens)
        text_token_count = self.config['max_text_length']
        text_tokens = numpy.ones((2, text_token_count), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, :len(tokens)] = tokens
        return text_tokens