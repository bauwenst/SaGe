# Copyright © 2023 Kensho Technologies, LLC

import time
import logging

import gensim.models
import numpy as np

from typing import Iterable
from pathlib import Path

from .tokeniser import SageTokenizer
from .util.iterables import FileAsStringIterable, TokenisedStringIterable
from .util.dataclasses import Word2VecParams

logger = logging.getLogger(__file__)


def get_embeddings(vocab_size: int, embeddings_folder: Path, partial_corpus: Iterable[str], sage_model: SageTokenizer, workers_number: int, word2vec_params: Word2VecParams) -> np.ndarray:
    logger.info(f"Training embeddings at vocab size {vocab_size}")
    embeddings_folder = Path(embeddings_folder)

    # is there an embedding of this size
    embeddings_filepath = embeddings_folder / f"embeddings_{vocab_size}.npy"
    if embeddings_filepath.exists():
        logger.info(f"Found trained embeddings. Loading it from {embeddings_filepath.as_posix()}")
        # context and target embeddings are the same so just keep one copy around
        embeddings = np.load(embeddings_filepath.as_posix())
    else:
        logger.info(f"Start training embeddings with Word2Vec...")
        start_time = time.time()
        embeddings = train_embeddings(sage_model, partial_corpus, workers_number, word2vec_params, embeddings_folder)
        logger.info(f"Embeddings time: {time.time() - start_time}")
        logger.info(f"Save embeddings to {embeddings_filepath.as_posix()}")
        np.save(embeddings_filepath.as_posix(), embeddings, allow_pickle=True)
    return embeddings


def train_embeddings(sage_model: SageTokenizer, partial_corpus: Iterable[str], workers: int, word2vec_params: Word2VecParams, embeddings_folder: Path) -> np.ndarray:
    tokenised_corpus = TokenisedStringIterable(partial_corpus, sage_model.pretokenize_and_tokenize_and_stringify)

    if isinstance(partial_corpus, FileAsStringIterable):  # GenSim is accelerated for file-stored corpora (https://github.com/RaRe-Technologies/gensim/releases/tag/3.6.0 and https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Any2Vec_Filebased.ipynb).
        gensim_file = embeddings_folder / f"gensim_{sage_model.vocab.size()}.txt"
        if gensim_file.exists():  # Caching
            logger.info(f"Tokenized corpus already exists at {gensim_file.as_posix()}")
        else:
            with open(gensim_file, "w", encoding="utf-8") as handle:
                for token_string in tokenised_corpus:
                    handle.write(token_string + "\n")
            logger.info(f"Tokenized data written at {gensim_file.as_posix()}")

        gensim_iterator = None
        gensim_file = gensim_file.as_posix()
    else:
        gensim_iterator = tokenised_corpus
        gensim_file = None

    word2vec_model = gensim.models.Word2Vec(
        corpus_file=gensim_file,
        sentences=gensim_iterator,

        vector_size=word2vec_params.D,
        negative=word2vec_params.N,
        alpha=word2vec_params.ALPHA,
        window=word2vec_params.window_size,
        min_count=word2vec_params.min_count,
        sg=word2vec_params.sg,
        workers=workers
    )
    embeddings = np.zeros(shape=(sage_model.vocab.size(), word2vec_params.D))

    for idx, token in sage_model.vocab.inv_str_vocab.items():
        if token in word2vec_model.wv.key_to_index.keys():
            embeddings[idx] = word2vec_model.wv[token]
        else:
            # some may not have made the min_count value, so will be missing
            # Embeddings not found for this token. Assign a random vector
            # doing this the same way as the old SaGe code
            embeddings[idx] = np.random.uniform(low=-0.5/word2vec_params.D, high=0.5/word2vec_params.D, size=(1, word2vec_params.D))

    # just return one copy that we'll use for both context and target embeddings
    return embeddings
