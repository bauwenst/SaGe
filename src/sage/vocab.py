from typing import Union, Iterable

from .util.pseudobytes import HFEncoding

hexstr = str
ParsableVocabulary = Union[ Iterable[Union[bytes,hexstr]], dict[Union[bytes,hexstr],int] ]


class SageVocab:

    def __init__(self):
        self.byte_vocab: dict[bytes, int]     = None
        self.inv_byte_vocab: dict[int, bytes] = None
        self.str_vocab: dict[str, int]        = None
        self.inv_str_vocab: dict[int, str]    = None

    def initialize(self, new_vocab: ParsableVocabulary, add_alphabet: bool):
        """
        Given an order list of bytes for the vocabulary, initialise all internal structures
        overwriting any previous values.
        """
        hfe = HFEncoding()

        # Set main bytes -> ID map, and make sure we always have all single bytes in vocabulary
        self.byte_vocab = SageVocab.parse_vocab(new_vocab)

        max_id = max(self.byte_vocab.values())
        for i in range(256):
            b = bytes([i])
            if b not in self.byte_vocab:
                if add_alphabet:
                    max_id += 1
                    self.byte_vocab[b] = max_id
                else:
                    raise Exception(f"missing byte {b}")

        # Inverted map ID -> bytes
        self.inv_byte_vocab = {v: k for (k, v) in self.byte_vocab.items()}
        # HuggingFace-equivalent of bytes -> ID
        self.str_vocab = {hfe.to_encoded(k): v for (k, v) in self.byte_vocab.items()}
        # ID -> HuggingFace-equivalent of bytes
        self.inv_str_vocab = {v: k for (k, v) in self.str_vocab.items()}

    def add_all_byte_ids(self, vocab: dict[int,float], score: float=1e400):
        """
        For each single byte, look up its id, then assign the given score to that id in the given dictionary.
        """
        for i in range(256):
            # what is the corresponding token id
            tid = self.byte_vocab[bytes([i])]
            # add that with a "good" score
            vocab[tid] = score

    @classmethod
    def parse_vocab(cls, raw_vocab: ParsableVocabulary) -> dict[bytes, int]:
        if isinstance(raw_vocab, dict):  # IDs have been pre-determined.
            parsed_vocab = {(bytes.fromhex(t) if isinstance(t, str) else t): i for t, i in raw_vocab.items()}
        else:
            parsed_vocab = {(bytes.fromhex(t) if isinstance(t, str) else t): i for i, t in enumerate(raw_vocab)}

        # Runtime type checking
        for t, i in parsed_vocab.items():
            assert isinstance(t, bytes)
            assert isinstance(i, int)

        return parsed_vocab

    def id_to_bytes(self, token_id: int) -> bytes:
        return self.inv_byte_vocab[token_id]

    def id_to_string(self, token_id: int) -> str:
        return self.inv_str_vocab[token_id]

    def print_tokens(self, ids: list[int]) -> list[bytes]:
        """
        Human readable for debugging
        """
        return [self.inv_byte_vocab[i] for i in ids]

    def size(self):
        return len(self.byte_vocab)
