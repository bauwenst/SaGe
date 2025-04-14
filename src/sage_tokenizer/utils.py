# Copyright © 2023 Kensho Technologies, LLC
from typing import Iterable, Tuple, List, Optional, Dict, Union, Generator
from abc import ABC, abstractmethod
from types import GeneratorType
from pathlib import Path

import sys
import json
import time
import random
import logging
import multiprocessing as mp

import numpy as np
from scipy.special import expit
from datasets import IterableDataset

from .model import SaGeTokenizer
from .paths import getDataFolder, getLogsFolder, getResultsFolder

# only log code outside of multiprocessing
# logger = logging.getLogger(__name__)


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

    def __init__(self, sentences: Iterable[str], tokeniser: SaGeTokenizer):
        self.sentences = sentences
        self.tokeniser = tokeniser

    def __iter__(self) -> Generator[str, None, None]:
        start = time.time()
        logging.info(f"Tokenizing corpus...")
        for i,s in enumerate(self.sentences):
            if i % 5_000 == 0:
                logging.info(f"\tTokenizing example {i}, time: {(time.time() - start):.2f} seconds")
            yield " ".join(self.tokeniser.tokenize_to_encoded_str(self.tokeniser.pretokenize(s)))


def write_vocab(vocab: Dict[bytes,int], filename: Path):
    """
    Dump the byte vocab to a file, encoded as hex characters inside this function.
    Saved in same order by index, so should preserve order.
    No special tokens are added.
    """
    byindex = sorted([(idx, token) for token, idx in vocab.items()])

    with open(filename, "w", encoding="utf-8") as f:
        for _, token in byindex:
            f.write(token.hex() + "\n")


def save_sorted_losses(sage_model: SaGeTokenizer, sorted_losses, target_vocab_size: int, vocab_folder: Path):
    vocab_folder = Path(vocab_folder)

    sorted_losses_filepath = vocab_folder / f"sorted_losses_before_{target_vocab_size}.txt"
    worst_500_filepath     = vocab_folder / f"worst_500_{target_vocab_size}.txt"
    best_500_filepath      = vocab_folder / f"best_500_{target_vocab_size}.txt"

    logging.info(f"Saving sorted losses to {sorted_losses_filepath.as_posix()}")
    write_sorted_losses_into_file(sorted_losses,   sorted_losses_filepath, sage_model)
    write_sorted_losses_into_file(sorted_losses[:500], worst_500_filepath, sage_model)
    write_sorted_losses_into_file(sorted_losses[-500:], best_500_filepath, sage_model)


def write_sorted_losses_into_file(sl: Iterable[Tuple[float,int]], filename: Path, sage_model: SaGeTokenizer):
    with open(filename, 'w', encoding="utf-8") as f:
        for loss, tid in sl:
            f.write(sage_model.id_to_encoded(tid) + "\t" + str(loss) + "\n")


def load_vocab(hex_string_source: TextSource) -> List[bytes]:
    """
    Read our hex formatted vocab file, return a list of bytes objects.
    Input file has one vocab word per line, each hex encoded.
    """
    return [bytes.fromhex(token) for token in textSourceToIterable(hex_string_source)]


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
    data = IterableDataset.from_generator(lambda: ({"text": s.strip().replace("\n", " ")} for s in corpus))  # It's actually "from_thingThatMakesGenerator", not "from_generator".
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


def compute_losses(losses, all_triples, embeddings):
    """
    function for computing losses given triple counts and embeddings
    losses : accumulate losses per ablated token, excluding the single byte ones, side effect this
    all_triples : triple values to aggregate into losses
    embeddings : embedding for each token
    """
    target_ids, context_ids, count = zip(*[(target_id, context_id, count) for (_, target_id, context_id), count in all_triples.items()])
    target_embeddings = np.array([embeddings[target_id] for target_id in target_ids])
    context_embeddings = np.array([embeddings[context_id] for context_id in context_ids])
    count = np.array(count)
    triples_loss = count * np.log(expit(np.einsum('ij,ij->i', target_embeddings, context_embeddings)))
    for idx, ((ablated_token_id, target_id, context_id), count) in enumerate(all_triples.items()):
        losses[ablated_token_id] = losses.get(ablated_token_id, 0.0) + triples_loss[idx]


def run_sage_parallel(embeddings: np.ndarray, partial_corpus: Iterable[str], sage_model: SaGeTokenizer, workers_number: int):
    logging.info(f"Splitting data into {workers_number} chunks.")
    # data_chunk_gen = divide_data_by_num(partial_corpus, workers_number)
    # data_chunk_gen = split_iterable_into_generators(partial_corpus, n=workers_number)
    data_divider = IterableDivideIntoNumber(partial_corpus, n_parts=workers_number)

    # these get aggregated over each chunk
    sage_losses = {}  # is token_id : loss
    overall_total_tokens = 0
    overall_total_triples = 0
    ablated_sizes = {}
    start_time = time.time()
    logging.info(f"Start spawning processes...")
    with mp.Pool(processes=workers_number) as pool:
        tasks = {}

        for tid in range(workers_number):
            res = pool.apply_async(sage_per_chunk, args=(tid, sage_model, embeddings, data_divider))
            tasks[res] = tid

        while tasks:  # Keep polling tasks for whether they are done.
            # Get newly finished tasks. These will be deleted afterwards.
            results_ready_list = []
            for res, tid in tasks.items():
                if res.ready():
                    results_ready_list.append((res, tid))

            # Process results for newly finished tasks.
            for res, tid in results_ready_list:
                losses, total_tokens, total_triples, ab_sizes = res.get()

                # just add these to totals/maxes
                overall_total_tokens += total_tokens
                overall_total_triples += total_triples

                # add to the overall tallys
                for k, v in losses.items():
                    sage_losses[k] = sage_losses.get(k, 0) + v

                # how many tokens needed to be examined
                for k, v in ab_sizes.items():
                    ablated_sizes[k] = ablated_sizes.get(k, 0) + v

                # all done with this,
                # can delete from tasks without messing up iteration over list
                del tasks[res]

                logging.info(f"task {tid} finished after {(time.time() - start_time):.2f} seconds. "
                             f"Tokens:{total_tokens}, triples:{total_triples}, active:{len(sage_losses)}")

            logging.info(f"Sleeping 1 second. Number of still running tasks: {len(tasks)}")
            time.sleep(1.0)
    return overall_total_tokens, overall_total_triples, sage_losses, ablated_sizes


def sage_per_chunk(tid: int, model: SaGeTokenizer, embeddings: np.ndarray, data: DataDivider, chunk_size: int=10_000):
    """
    function that runs sage on each chunk of data (in parallelization)
    note: this is called from multiprocessing, so use print rather than logging
    """
    print(f"Starting chunk {tid}.")
    start_time = time.time()

    # accumulate over all the data
    losses = {}

    # these accumulate over each size
    triples = {}
    ablated_sizes = {}
    total_tokens = 0
    total_triples = 0
    total_fs_time = 0.0
    total_cl_time = 0.0

    fs_start = time.time()
    n_examples_seen = 0
    for i, sentence in enumerate(data.getPart(tid)):
        n_examples_seen += 1
        total_tokens += model.fast_sage(model.pretokenize(sentence), triples, ablated_sizes)

        # if filled up chunk, compute the losses to free up memory
        if i > 0 and i % chunk_size == 0:
            # take the total time here over all calls
            fs_time = time.time() - fs_start
            total_fs_time += fs_time
            # reinitialize fs_start
            fs_start = time.time()

            cl_start = time.time()
            compute_losses(losses, triples, embeddings)
            cl_time = time.time() - cl_start
            total_cl_time += cl_time

            print(f"fast_sage {tid}, row {i+1}, "
                  f"fs_time: {fs_time:.2f}, cl_time: {cl_time:.2f}, "
                  f"triples: {len(triples)}, tokens: {total_tokens}")

            # total these up
            total_triples += len(triples)

            # zero out the triples from this chunksize lines
            triples = {}

    # compute for final partial chunk
    compute_losses(losses, triples, embeddings)
    total_triples += len(triples)

    # the triples can get quite large, so to avoid merging these
    # dict values, let's compute the losses in parallel too
    print(f"final fast_sage {tid}, row {n_examples_seen} of {n_examples_seen}, "
          f"fs_time: {total_fs_time:.2f}, cl_time: {total_cl_time:.2f}, time: {(time.time() - start_time):.2f}, "
          f"triples: {len(triples)}, tokens: {total_tokens}")

    # Extra negative sign for equation (1) in SaGe paper
    # track number in cache too
    losses = {k: -v for k, v in losses.items()}

    return losses, total_tokens, total_triples, ablated_sizes


def init_logger(experiment_name: str, do_stdout_too: bool=False):
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_filename = getLogsFolder() / f"{experiment_name}_{timestamp_str}.log"
    logging.basicConfig(
        handlers=[logging.FileHandler(log_filename.as_posix())] + do_stdout_too*[logging.StreamHandler(sys.stdout)],
        format="[%(asctime)s @ %(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    print(f"{'All' if not do_stdout_too else 'A copy of the'} logs will be stored at {log_filename.as_posix()}")


def get_output_folder(experiment_name: str) -> Tuple[Path, Path, Path]:
    results_path = getResultsFolder() / experiment_name
    results_path.mkdir(exist_ok=True, parents=True)

    vocab_folder = results_path / "sage_vocabs"
    vocab_folder.mkdir(exist_ok=True)

    stats_folder = results_path / "stats"
    stats_folder.mkdir(exist_ok=True)

    embeddings_folder = results_path / "embeddings"
    embeddings_folder.mkdir(exist_ok=True)

    return embeddings_folder, stats_folder, vocab_folder


def set_random_seed(experiment_name: str, random_seed: int):
    # Log seed
    seed_filepath = getResultsFolder() / experiment_name / "seed.txt"
    with open(seed_filepath, "w+") as f:
        f.write(str(random_seed))

    # Set seed
    random.seed(random_seed)
    np.random.seed(random_seed)


def save_stats(stats: dict, stats_folder: Path, target_vocab_size: int):
    stats_folder = Path(stats_folder)

    stats_filename = stats_folder / f"stats_{target_vocab_size}.json"
    logging.info(f"Saving stats to {stats_filename.as_posix()}")
    with open(stats_filename, "w") as f:
        json.dump(stats, f, indent=2)  # pretty print a bit
        f.write("\n")
