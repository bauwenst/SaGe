from typing import Iterable, Tuple, List, Optional, Dict, Union, Generator, Callable
from abc import ABC, abstractmethod
from pathlib import Path
from datasets import IterableDataset

import logging
import time

from .paths import getDataFolder

TextSource = Union[ Iterable[str], Union[str,Path] ]
Reiterable = Iterable  # There isn't an actual type that represents "all iterables that DON'T have a next() method", so we fake it in the type annotations.


def textSourceToIterable(source: TextSource) -> Iterable[str]:
    if isinstance(source, (str, Path)):
        return FileAsStringIterable(Path(source))
    else:
        return source


class FileAsStringIterable:
    """
    Stores a file path and can return infinitely many generators (rather than being a generator).
    """

    def __init__(self, text_file: Path):
        if not text_file.exists():
            raise FileNotFoundError(f'Missing file: {text_file.as_posix()}')

        self.path = text_file

    def __iter__(self) -> Generator[str, None, None]:
        logging.info(f"Loading contents from {self.path.as_posix()}")
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                yield line


class DictIterableAsStringIterable:

    def __init__(self, iterable_dataset: IterableDataset, field: str="text"):
        self.dict_iterable = iterable_dataset
        self.field = field

    def __iter__(self) -> Generator[str, None, None]:
        for d in self.dict_iterable:
            yield d[self.field]


class TokenisedStringIterable:
    """
    GenSim expects a corpus consisting of whitespace-separated tokens, so this takes each sentences, pretokenises it,
    tokenises it, and concatenates the resulting tokens with spaces.
    """

    def __init__(self, sentences: Reiterable[str], tokeniser: Callable[[str],list[str]]):
        self.sentences = sentences
        self.tokeniser = tokeniser

    def __iter__(self) -> Generator[str, None, None]:
        start = time.time()
        logging.info(f"Tokenizing corpus...")
        for i,s in enumerate(self.sentences):
            if i % 5_000 == 0:
                logging.info(f"\tTokenizing example {i}, time: {(time.time() - start):.2f} seconds")
            yield " ".join(self.tokeniser(s))


def load_corpus(corpus: TextSource, n_corpus_examples: Optional[int], cache_name_or_path: Union[str,Path], seed: int) -> Iterable[str]:
    if isinstance(cache_name_or_path, Path):
        do_cache = True
        cache_path = cache_name_or_path.with_suffix(".txt")
    else:
        do_cache = len(cache_name_or_path) > 0
        if do_cache:
            if "/" in cache_name_or_path:  # User wants a very specific path, e.g. because he's running on an HPC and has a separate path for storage.
                cache_path = Path(cache_name_or_path).with_suffix(".txt")
            else:
                cache_path = getDataFolder() / f"{cache_name_or_path}_{n_corpus_examples if n_corpus_examples else 'full'}_seed{seed}.txt"
        else:
            cache_path = None

    if do_cache and cache_path.exists():
        logging.info(f"Found pre-existing partial corpus.")
        # start = time.time()
        # with open(cache_path, "r") as handle:
        #     partial_corpus = handle.readlines()
        # logging.info(f"Size of Corpus: {len(partial_corpus)}, time: {(time.time() - start):.2f}")
        return FileAsStringIterable(cache_path)

    corpus = textSourceToIterable(corpus)
    data = IterableDataset.from_generator(generator=corpusToExamples, gen_kwargs={"corpus": corpus})  # It's actually "from_thingThatMakesGenerator", not "from_generator".
    # data = IterableDataset.from_generator(lambda: ({"text": s.strip().replace("\n", " ")} for s in corpus))  # It's actually "from_thingThatMakesGenerator", not "from_generator".
    data = data.shuffle(buffer_size=1_000_000, seed=seed)
    data = data.take(n_corpus_examples) if n_corpus_examples else data
    data = DictIterableAsStringIterable(data)

    if do_cache:
        # read_start = time.time()
        # with open(corpus_filepath, "r") as handle:
        #     corpus = handle.readlines()
        #     logging.info(f"Loading from Original Corpus. Number of lines: {len(corpus)}")
        #
        # random.shuffle(corpus)
        # logging.info(f"Original Corpus read and shuffled. Time: {(time.time() - read_start):.2f}")
        write_start_time = time.time()
        # partial_corpus = corpus[:n_corpus_examples]
        # cache_path = getDataFolder() / f"{corpus_filepath.stem}_{n_corpus_examples}.txt"

        n_corpus_examples_actual = 0
        with open(cache_path, "w+", encoding="utf-8") as handle:
            for string in data:
                handle.write(string + "\n")
                n_corpus_examples_actual += 1

        logging.info(f"Partial corpus saved at {cache_path.as_posix()}. "
                     f"Number of lines: {n_corpus_examples_actual}, "
                     f"time: {(time.time() - write_start_time):.2f}")
        return FileAsStringIterable(cache_path)
    else:
        return data


def corpusToExamples(corpus: Iterable[str]) -> Iterable[Dict[str,str]]:
    return ({"text": s.strip().replace("\n", " ")} for s in corpus)


def hexStringsToBytes(hex_strings: TextSource) -> list[bytes]:
    return [bytes.fromhex(token) for token in textSourceToIterable(hex_strings)]


class DataDivider(ABC):
    @abstractmethod
    def getPart(self, i: int) -> Iterable[str]:
        pass


class ListDivider(DataDivider):
    def __init__(self, list_to_divide: List[str]):
        self.list_to_divide = list_to_divide


class ListDivideIntoSize(ListDivider):

    def __init__(self, list_to_divide: List[str], part_size: int):
        super().__init__(list_to_divide)
        self.part_size = part_size

    def getPart(self, i: int) -> Iterable[str]:
        return self.list_to_divide[i*self.part_size:(i+1)*self.part_size]


class ListDivideIntoNumber(ListDivideIntoSize):

    def __init__(self, list_to_divide: List[str], n_parts: int):
        super().__init__(list_to_divide, part_size=len(list_to_divide) // n_parts)


class IterableDivideIntoNumber(DataDivider):
    """
    If the old iterable goes
        a b c d e f g h
    and we split into 3 iterables, then they go
        a d g

        b e h

        c f
    """

    def __init__(self, reiterable: Reiterable[str], n_parts: int):
        self.reiterable = reiterable
        self.n_parts = n_parts

    def getPart(self, i: int) -> Iterable[str]:
        for example_idx, example in enumerate(self.reiterable):
            if (example_idx - i) % self.n_parts == 0:  # example_idx-i is equivalent to example_idx + (n_parts-i) in modular arithmetic, so when example_idx == i you get n_parts % n_parts == 0.
                yield example
