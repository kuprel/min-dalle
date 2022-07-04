from math import inf
from typing import List, Tuple

class TextTokenizer:
    def __init__(self, vocab: dict, merges: List[str]):
        self.token_from_subword = vocab
        pairs = [tuple(pair.split()) for pair in merges]
        self.rank_from_pair = dict(zip(pairs, range(len(pairs))))

    def tokenize(self, text: str, is_verbose: bool = False) -> List[int]:
        sep_token = self.token_from_subword['</s>']
        cls_token = self.token_from_subword['<s>']
        unk_token = self.token_from_subword['<unk>']
        text = text.lower().encode("ascii", errors="ignore").decode()
        tokens = [
            self.token_from_subword.get(subword, unk_token)
            for word in text.split(" ") if len(word) > 0
            for subword in self.get_byte_pair_encoding(word, is_verbose)
        ]
        return [cls_token] + tokens + [sep_token]

    def get_byte_pair_encoding(self, word: str, is_verbose: bool) -> List[str]:
        def get_pair_rank(pair: Tuple[str, str]) -> int:
            return self.rank_from_pair.get(pair, inf)

        subwords = [chr(ord(" ") + 256)] + list(word)
        while len(subwords) > 1:
            pairs = list(zip(subwords[:-1], subwords[1:]))
            pair_to_merge = min(pairs, key=get_pair_rank)
            if pair_to_merge not in self.rank_from_pair: break
            i = pairs.index(pair_to_merge)
            subwords = (
                (subwords[:i] if i > 0 else []) + 
                [subwords[i] + subwords[i + 1]] + 
                (subwords[i + 2:] if i + 2 < len(subwords) else [])
            )

        if is_verbose: print(subwords)
        return subwords