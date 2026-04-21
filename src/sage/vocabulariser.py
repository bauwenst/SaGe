# Copyright © 2023 Kensho Technologies, LLC
from typing import List, Union, Optional
from pathlib import Path

import logging

from .embeddings import get_embeddings
from .tokeniser import SageTokenizer
from .loss import run_sage_parallel, save_sorted_losses
from .util.paths import init_logger, get_output_folder, set_random_seed, write_vocab
from .util.iterables import load_corpus, TextSource, hexStringsToBytes
from .util.paths import save_stats
from .util.dataclasses import Word2VecParams

logger = logging.getLogger(__name__)


class SaGe:

    def __init__(self, full_vocab_schedule: List[int], embeddings_schedule: List[int],
                 max_len: int=16, workers_number: int=1, random_seed: int=692653,
                 word2vec_d: int=50, word2vec_n: int=15, word2vec_alpha: float=0.025, word2vec_window_size: int=5, word2vec_min_count: int=1, word2vec_sg: bool=True):
        self.full_vocab_schedule = full_vocab_schedule
        self.embeddings_schedule = embeddings_schedule
        self.max_len = max_len
        self.workers_number = workers_number
        self.random_seed = random_seed
        self.word2vec_params = Word2VecParams(
            D=word2vec_d,
            N=word2vec_n,
            ALPHA=word2vec_alpha,
            window_size=word2vec_window_size,
            min_count=word2vec_min_count,
            sg=int(word2vec_sg)  # 1 uses skip-gram, 0 uses CBoW.
        )

    def build(self, experiment_name: str,
              initial_vocabulary: TextSource,
              corpus: TextSource, k_corpus_examples: Optional[int]=1_000, corpus_cache: Union[str,Path]="",
              do_log_stdout: bool=False) -> Path:
        """
        :param experiment_name: Prefix for the outputs of this run.
        :param initial_vocabulary: The set of subwords to start from. The subwords are expected to be strings obtained
                                   by converting the actual subword strings to UTF-8 bytes and then converting those to
                                   hexadecimal.
        :param corpus: Either an iterable of string examples, or a text file. If the latter, every newline starts a new example.
                       Note: must NOT be an iteraTOR, i.e. it must be possible to iterate over this multiple times without it being consumed afterwards.
        :param k_corpus_examples: How many k's (thousands) of examples to sample from the corpus.
                                  If None or 0, the entire corpus is used.
        :param corpus_cache: If an empty string, examples are streamed from the given corpus directly (after shuffling and
                             truncating) and GenSim's slower implementation is used.
                             Else, all used examples from the corpus will be cached into a text file. The file will be
                             located under PATH_SAGE/data/ if the given value is a string with no slashes, otherwise
                             under the specific path it points to. This file may be huge, but GenSim runs much faster
                             when it runs on a file rather than a stream.
        """
        assert k_corpus_examples is None or k_corpus_examples >= 0

        init_logger(experiment_name, do_stdout_too=do_log_stdout)
        logger.info(f"Start experiment {experiment_name}")
        logger.info(f"Process will use up to {self.workers_number} worker threads.")
        logger.info("Getting output directories")
        embeddings_folder, stats_folder, vocab_folder = get_output_folder(experiment_name)
        logger.info("Setting random seed")
        set_random_seed(experiment_name, self.random_seed)
        logger.info(f"Loading initial vocabulary...")
        byte_vocab = hexStringsToBytes(initial_vocabulary)
        logger.info(f"Finished loading initial vocabulary. Vocabulary size: {len(byte_vocab)}")

        actual_max_len = max([len(v) for v in byte_vocab])
        if self.max_len != actual_max_len:
            logger.warning(f"Note that the max_len parameter value {self.max_len} doesn't match actual max {actual_max_len}")

        logger.info("Initializing tokenizer")
        sage_model = SageTokenizer(byte_vocab, max_len=self.max_len)
        logger.info(f"Loading corpus...")
        partial_corpus = load_corpus(corpus, n_corpus_examples=1000*k_corpus_examples if k_corpus_examples else None, cache_name_or_path=corpus_cache, seed=self.random_seed)
        logger.info("Starting the training loop")
        vocab_schedule = self.full_vocab_schedule

        if not len(vocab_schedule) >= 2:
            raise Exception("Vocabulary schedule must contain more than 2 vocabulary sizes!")

        vocab_schedule.sort(reverse=True)  # largest first
        logger.info(f"Initial vocab_schedule value is {vocab_schedule[0]} vs. actual size {sage_model.vocab.size()}")

        embedding_sizes = set(self.embeddings_schedule)

        # skipping the initial vocab size here
        i = 0
        compute_embeddings_at_size = vocab_schedule[0]
        embeddings_up_to_date = False
        embeddings = None

        path_full_vocab: Path = None
        ever_active_types: set[int] = set()
        while i < len(vocab_schedule) - 1:  # -1 because the last vocab size is only a target, but never current.
            current_step_vocab_size = vocab_schedule[i]  # this will be the label used for files
            target_vocab_size       = vocab_schedule[i+1]
            actual_vocab_size = sage_model.vocab.size()
            logger.info(f"\nRound {i+1} - Start: "
                        f"\n\tCurrent step vocabulary size: {current_step_vocab_size}"
                        f"\n\tTarget vocabulary size: {target_vocab_size}"
                        f"\n\tActual vocabulary size: {actual_vocab_size}")

            if vocab_schedule[i] in embedding_sizes:
                compute_embeddings_at_size = vocab_schedule[i]
                embeddings_up_to_date = False

            if actual_vocab_size <= target_vocab_size:
                logger.info(f"Actual vocab is already smaller than target. SaGe won't be used. Continuing to next iteration.")
                i += 1
                continue

            if not embeddings_up_to_date:
                embeddings = get_embeddings(compute_embeddings_at_size, embeddings_folder, partial_corpus, sage_model,
                                            self.workers_number, self.word2vec_params)
                embeddings_up_to_date = True

            # call sage in parallel
            logger.info(f"Sage started.")
            total_tokens, total_triples, token_to_losses, ablated_sizes = run_sage_parallel(embeddings,
                                                                                            partial_corpus,
                                                                                            sage_model,
                                                                                            self.workers_number)
            logger.info(f"Sage finished. total tokens: {total_tokens}, total triplets: {total_triples}")

            # token_to_losses won't include any single byte tokens, but we want to keep those around
            # so lets just add them with large scores, so they stay around
            current_active_vocab_size_before_ensuring_alphabet = len(token_to_losses)
            sage_model.vocab.add_all_byte_ids(token_to_losses, score=1e6)
            logger.info(f"Adding single bytes to vocab. Size before: {current_active_vocab_size_before_ensuring_alphabet}, "
                        f"size after: {len(token_to_losses)}")

            # If a token doesn't appear in token_to_losses, it didn't participate in the tokenization.
            #
            # There are three types of vocabularies in SaGe: the actual vocab, the active vocab, and the inactive vocab.
            #   - the actual vocab is all types that have not been pruned.
            #   - the active vocab is all types that have been seen in the latest tokenisation of the corpus. These are the types we can actually say anything about.
            #   - the inactive vocab is all types that are actual but not active. We know nothing about these.
            # When we prune to get to the next target size, we calculate how many types we need to drop, and then we
            # take away types from the actual vocab by looking at the least impactful ACTIVE types.
            # This means that when a type never appears in the corpus, it will actually be preserved since it is never active.
            current_active_vocab_size = len(token_to_losses)
            current_inactive_vocab_size = actual_vocab_size - len(token_to_losses)
            logger.info(f"Actual vocab size: {actual_vocab_size}, "
                        f"Target vocab size: {target_vocab_size}, "
                        f"Active vocab size (before pruning): {current_active_vocab_size}, "
                        f"Inactive vocab size: {current_inactive_vocab_size}")

            ever_active_types.update(token_to_losses.keys())

            # how many of the losses are negative
            neg_loss  = len([loss for loss in token_to_losses.values() if loss < 0.0])
            zero_loss = len([loss for loss in token_to_losses.values() if loss == 0.0])
            pos_loss  = len([loss for loss in token_to_losses.values() if loss > 0.0])
            logger.info(f"Negative losses: {neg_loss}, zero losses: {zero_loss}, positive losses: {pos_loss}")

            # in case the active vocab we found is actually smaller than the target vocab,
            # change the target to the next one, until it's smaller than the vocab we found,
            # so the ablation part will actually do something
            while current_active_vocab_size <= target_vocab_size:
                logger.info(f"Active vocab size is {current_active_vocab_size}, which is "
                            f"smaller than target {target_vocab_size}. Moving to next target_vocab_size"
                            f"\n\n(Round increased to {i + 1})\n")
                i += 1
                if i == len(vocab_schedule) - 1:
                    logger.info("The active vocab is so small that we don't even have enough information to prune any type. Terminating here.")
                    target_vocab_size = current_active_vocab_size
                    break
                target_vocab_size = vocab_schedule[i+1]
                logger.info(f"New target_vocab_size: {target_vocab_size}")

            n_types_to_prune = current_active_vocab_size - target_vocab_size  # TODO: This formula is strange, because the actual vocab size may be much larger than the target vocab size after pruning this amount of types.
            assert n_types_to_prune >= 0
            logger.info(f"Types to prune: {n_types_to_prune}")

            ######################
            # do the ablation
            ######################
            # we want to drop the smallest (negative) values
            # these are the ones with the largest decrease in likelihood from dropping the ablated token
            sorted_losses = list(sorted([(loss, tid) for (tid, loss) in token_to_losses.items()]))
            save_sorted_losses(sage_model.vocab, sorted_losses, target_vocab_size, vocab_folder)

            save_stats({
                "current_step_vocab_size": current_step_vocab_size,
                "total_tokens": total_tokens,
                "total_triples": total_triples,
                "current_active_vocab_size": current_active_vocab_size,
                "current_inactive_vocab_size": current_inactive_vocab_size,
                "neg_loss": neg_loss,
                "zero_loss": zero_loss,
                "pos_loss": pos_loss,
                "target_vocab_size": target_vocab_size,
                "num_tokens_to_prune": n_types_to_prune,
                "ablated_sizes": ablated_sizes,
            }, stats_folder, target_vocab_size)

            # these are the tokens to be removed
            types_to_prune = {sage_model.vocab.id_to_bytes(tid) for (loss, tid) in sorted_losses[:n_types_to_prune]}
            # double check there are no single bytes tokens to prune here
            assert len([token for token in types_to_prune if len(token) == 1]) == 0

            # our overall vocabulary after pruning
            target_vocab = {tok: tid for tok, tid in sage_model.vocab.byte_vocab.items()
                            if tok not in types_to_prune}
            path_full_vocab = vocab_folder / f"sage_vocab_{target_vocab_size}.vocab"
            logger.info(f"Saving intermediate vocab of size {len(target_vocab)} to {path_full_vocab.as_posix()}")
            write_vocab(target_vocab, path_full_vocab)

            # our active vocabulary *after* pruning; is active if it has an entry in token_to_losses
            active_vocab = {tok: tid for tok, tid in sage_model.vocab.byte_vocab.items()
                            if tid in token_to_losses and tok not in types_to_prune}
            path_active_vocab = vocab_folder / f"active_vocab_{target_vocab_size}.vocab"
            logger.info(f"Saving active vocab of size {len(active_vocab)} to {path_active_vocab.as_posix()}")
            write_vocab(active_vocab, path_active_vocab)

            # the deleted items; saved for analysis, with the original size
            deleted_vocab = {tok: tid for tok, tid in sage_model.vocab.byte_vocab.items()
                             if tok in types_to_prune}
            path_pruned_vocab = vocab_folder / f"deleted_vocab_{target_vocab_size}.vocab"
            logger.info(f"Saving deleted vocab of size {len(deleted_vocab)} to {path_pruned_vocab.as_posix()}")
            write_vocab(deleted_vocab, path_pruned_vocab)

            # save a version of the actual vocabulary that excludes tokens that were never seen
            effective_vocab = {typ: id for typ, id in target_vocab.items()
                               if id in ever_active_types}
            path_effective_vocab = vocab_folder / f"effective_vocab_{target_vocab_size}.vocab"
            logger.info(f"Saving effective vocab of size {len(effective_vocab)} to {path_effective_vocab.as_posix()}")
            write_vocab(effective_vocab, path_effective_vocab)

            # now update the internal state of sage_model to use the new smaller vocab
            # pass in list of bytes keys, which keep insertion order
            sage_model.vocab.initialize(list(target_vocab.keys()), add_alphabet=True)

            logger.info(f"\nRound {i} - End: "
                        f"\n\tCurrent step vocab size: {current_step_vocab_size}, "
                        f"\n\tTarget vocab size: {target_vocab_size}, "
                        f"\n\tActive vocab size (after pruning): {len(active_vocab)}")

            # advance to next smaller size
            i += 1

        assert path_full_vocab is not None  # Not possible because the while is over len(schedule)-1 and the schedule has at least 2 points.
        return path_full_vocab
