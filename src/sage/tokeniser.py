# Copyright © 2023 Kensho Technologies, LLC
from typing import Union

from .vocab import SageVocab, ParsableVocabulary

Tokenizable = Union[str,bytes]


class SageTokenizer:
    """
    Byte-based left-to-right greedy tokeniser.
    Basically just a SageVocab with a
    """

    def __init__(self, initial_vocabulary: ParsableVocabulary, add_alphabet: bool=False, max_len: int=16):
        self.vocab = SageVocab()
        self.vocab.initialize(initial_vocabulary, add_alphabet=add_alphabet)
        self.max_len = max_len

    def pretokenize(self, sentence: Tokenizable) -> bytes:
        try:  # Most frequent use-case is tokenising a string, so this is much faster than `if isinstance(sentence, str)`.
            return sentence.encode("utf-8")
        except:
            return sentence

    def tokenize(self, sentence: bytes) -> list[int]:
        """
        Split the gives sentence into tokens and convert them to IDs.
        """
        data = []
        i = 0
        while i < len(sentence):  # Iterate through the sentence input
            for j in range(self.max_len, 0, -1):  # Find the longest possible token
                tok = sentence[i:i + j]
                if tok in self.vocab.byte_vocab:
                    # Only add token_id to results
                    data.append(self.vocab.byte_vocab[tok])
                    i += j  # advance to next token
                    break  # the for loop
        return data

    def pretokenize_and_tokenize(self, sentence: Tokenizable) -> list[int]:
        return self.tokenize(self.pretokenize(sentence))

    def pretokenize_and_tokenize_and_stringify(self, sentence: Tokenizable) -> list[str]:
        return [self.vocab.inv_str_vocab[id] for id in self.pretokenize_and_tokenize(sentence)]
