from pathlib import Path
import time
import logging

import numpy as np
from scipy.special import expit
import multiprocessing as mp

from .util.iterables import Iterable, IterableDivideIntoNumber, DataDivider
from .tokeniser import SageTokenizer
from .vocab import SageVocab


def compute_losses(losses: dict[int,float], all_triples: dict[tuple[int,int,int], int], embeddings):
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


def run_sage_parallel(embeddings: np.ndarray, partial_corpus: Iterable[str], sage_model: SageTokenizer, workers_number: int) -> tuple[int, int, dict[int,float], dict[int,int]]:
    logging.info(f"Splitting data into {workers_number} chunks.")
    # data_chunk_gen = divide_data_by_num(partial_corpus, workers_number)
    # data_chunk_gen = split_iterable_into_generators(partial_corpus, n=workers_number)
    data_divider = IterableDivideIntoNumber(partial_corpus, n_parts=workers_number)

    # these get aggregated over each chunk
    sage_losses: dict[int,float] = {}  # is token_id : loss
    overall_total_tokens = 0
    overall_total_triples = 0
    ablated_sizes: dict[int,int] = {}
    start_time = time.time()
    latest_print = time.time()
    logging.info(f"Start spawning processes...")
    with mp.Pool(processes=workers_number) as pool:
        tasks = {}

        for tid in range(workers_number):
            res = pool.apply_async(run_sage, args=(tid, sage_model, embeddings, data_divider))
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

            if time.time() - latest_print > 60:
                logging.info(f"Still polling {len(tasks)} incomplete tasks.")
                latest_print = time.time()
            time.sleep(1.0)
    return overall_total_tokens, overall_total_triples, sage_losses, ablated_sizes


def run_sage(tid: int, model: SageTokenizer, embeddings: np.ndarray, data: DataDivider, chunk_size: int=10000, progress_size: int=1000):
    """
    function that runs sage on each chunk of data (in parallelization)
    note: this is called from multiprocessing, so (1) use print rather than logging and (2) all results are returned rather than being in-place.
    """
    print(f"Running SaGe on part {tid} of the dataset.")
    start_time = time.time()

    # accumulate over all the data
    losses = {}

    # these accumulate over each size
    triples: dict[tuple[int,int,int], int] = {}
    ablated_sizes: dict[int,int] = {}
    total_tokens = 0
    total_triples = 0
    total_fastsage_time = 0.0
    total_computeloss_time = 0.0

    start_fastsage = time.time()
    n_examples_seen = 0
    for sentence in data.getPart(tid):
        total_tokens += fast_sage(model.pretokenize(sentence), triples, ablated_sizes, model.vocab, model.max_len)
        n_examples_seen += 1

        if n_examples_seen % progress_size == 0:
            print(f"SaGe thread {tid} has processed {n_examples_seen} rows. (Memory is freed every {chunk_size} rows.)")

        # if filled up chunk, compute the losses to free up memory
        if n_examples_seen % chunk_size == 0:
            # Total time over all calls to fast_sage.
            duration_fastsage = time.time() - start_fastsage
            total_fastsage_time += duration_fastsage
            start_fastsage = time.time()  # TODO: Are we sure we don't want this AFTER the call to compute_losses?

            start_computeloss = time.time()
            compute_losses(losses, triples, embeddings)
            duration_computeloss = time.time() - start_computeloss
            total_computeloss_time += duration_computeloss

            print(f"SaGe {tid} finished a chunk after {n_examples_seen} rows."
                  f"\n\tTime spent on fast_sage in this chunk: {duration_fastsage:.2f}"
                  f"\n\tTime spent on compute_losses in this chunk: {duration_computeloss:.2f}"
                  f"\n\tTriples in this chunk: {len(triples)}. "
                  f"\n\tTokens so far: {total_tokens}. Triples so far: {total_triples}")

            # total these up
            total_triples += len(triples)

            # zero out the triples from this chunksize lines
            triples = {}

    # compute for final partial chunk
    if triples:
        compute_losses(losses, triples, embeddings)
        total_triples += len(triples)

    # the triples can get quite large, so to avoid merging these
    # dict values, let's compute the losses in parallel too
    print(f"final fast_sage {tid}, row {n_examples_seen} of {n_examples_seen}, "
          f"fs_time: {total_fastsage_time:.2f}, cl_time: {total_computeloss_time:.2f}, time: {(time.time() - start_time):.2f}, "
          f"triples: {len(triples)}, tokens: {total_tokens}")

    # Extra negative sign for equation (1) in SaGe paper
    # track number in cache too
    losses = {k: -v for k, v in losses.items()}

    return losses, total_tokens, total_triples, ablated_sizes


def do_triples(combined, pad: int, padleft: int, padright: int, cur_id: int, sign: int, triples: dict[tuple[int,int,int], int]):
    """
    Add the appropriate (t,v,v') triples to our dictionary where t, v, and v' are all int indices.
    """
    # where the right padding starts
    right_ind = len(combined) - padright

    # iterate over the targets
    # note that the padding elements now have different contexts in
    # center section, so need to let them be targets too
    for t, target in enumerate(combined):
        # the contexts, need pad here not padleft or padright,
        # since some context may be within the combined
        for c in range(t - pad, t + pad + 1):
            # context is in range and distinct from target
            # ignore the case where both c and t are in padding since that cancels
            if c >= 0 and c != t and c < len(combined) and \
                    ((c >= padleft and c < right_ind) or (t >= padleft and t < right_ind)):
                trip = (cur_id, target, combined[c])
                # add sign to the triples
                triples[trip] = triples.get(trip, 0) + sign


def fast_sage(sentence: bytes, triples: dict[tuple[int,int,int], int], ablated_sizes: dict[int,int],
              vocab: SageVocab, max_len: int, pad: int=2,
              verbose: bool=False) -> int:
    """
    Tokenize the sentence `sent`, add to the counts in the triples dict tracking the (cur_id,t,c) for the ablated
    token cur_id, with target token t and context token c. Also updates the statistics in ablated_sizes.
    Returns the total_tokens from tokenizing `sent`.
    """
    n = len(sentence)
    if n == 0:
        return 0

    # if you use np.array here, remember to fix concatenation below
    ids           = []
    start_indices = []
    widths        = []
    i = 0
    while i < len(sentence):  # Iterate through the sentence input
        for j in range(max_len, 0, -1):  # Find the longest possible token
            span = sentence[i:i + j]
            if span in vocab.byte_vocab:
                ids.append(vocab.byte_vocab[span])
                start_indices.append(i)
                widths.append(len(span))
                i += j  # advance to next token
                break  # the for loop

    # note, these are arrays over the tokens so len(values) < n
    total_tokens = len(ids)
    maximum_tokens_found = 0

    # have a constant time lookup on whether we're at a token on the base tokenization
    # if >= 0, is the index of the token in ids or widths
    on_base = np.zeros(n, dtype=int) - 1
    for j, si in enumerate(start_indices):
        on_base[si] = j
    # now we can just produce our ablated tokenizations
    # quite efficiently
    for loc, (cur_id, start_index, width) in enumerate(zip(ids, start_indices, widths)):
        # skip single bytes
        if width > 1:
            ablated_tokenization = []

            # find the next token with width-1 or less
            # starting at start_index
            i = start_index
            for j in range(width - 1, 0, -1):
                tok = sentence[i:i + j]
                if tok in vocab.byte_vocab:
                    ablated_tokenization.append(vocab.byte_vocab[tok])  # keep the ids
                    i += j  # advance to next token
                    break  # the for loop

            # now extend as normal until we get back on the old path
            while i < n:
                for j in range(min(max_len, n - i), 0, -1):
                    tok = sentence[i:i + j]
                    if tok in vocab.byte_vocab:
                        ablated_tokenization.append(vocab.byte_vocab[tok])
                        i += j  # advance to next token
                        break  # the for loop

                # we never got back on the path, so set beyond to n
                if i >= n:
                    beyond = n
                    break

                # we get to a spot on the current longest path
                # we're back to the old tokenization, set beyond accordingly
                if on_base[i] != -1:
                    beyond = on_base[i]
                    break

            if verbose:
                print(vocab.print_tokens(ablated_tokenization))

            # track how many tokens were required for the ablation
            lat = len(ablated_tokenization)
            ablated_sizes[lat] = ablated_sizes.get(lat, 0) + 1
            maximum_tokens_found = max(maximum_tokens_found, lat)

            # note: on_base[i] is one beyond the last difference
            base_tok = ids[loc:beyond]
            if verbose:
                print(vocab.print_tokens(base_tok))

            # can we do any padding on left or right
            padleft = min(pad, loc)
            padright = min(pad, total_tokens - beyond)
            left_pad = ids[loc - padleft:loc]
            # print(print_tokens(left_pad))
            # note: beyond is one beyond the last difference
            right_pad = ids[beyond:beyond + padright]
            # print(print_tokens(right_pad))

            # combine with the padding, and work out the context triples
            combined_ab = left_pad + ablated_tokenization + right_pad
            do_triples(combined_ab, pad, padleft, padright, cur_id, 1, triples)

            # and same for the base tokenization
            combined_base = left_pad + base_tok + right_pad
            do_triples(combined_base, pad, padleft, padright, cur_id, -1, triples)

            if verbose:
                print("base:", vocab.print_tokens(left_pad), vocab.print_tokens(base_tok), vocab.print_tokens(right_pad))
                print("ab:  ", vocab.print_tokens(left_pad), vocab.print_tokens(ablated_tokenization), vocab.print_tokens(right_pad))
                print("comb base:", vocab.print_tokens(combined_base))
                print("comb ab:", vocab.print_tokens(combined_ab))
                print()

    # log some of these
    if maximum_tokens_found > 200:
        # remember to convert from bytes
        print("long max_len:", maximum_tokens_found, '"' + sentence.decode('utf-8') + '"')

    return total_tokens


def save_sorted_losses(vocab: SageVocab, sorted_losses: list[tuple[float,int]], target_vocab_size: int, vocab_folder: Path):
    vocab_folder = Path(vocab_folder)

    sorted_losses_filepath = vocab_folder / f"sorted_losses_before_{target_vocab_size}.txt"
    worst_500_filepath     = vocab_folder / f"worst_500_{target_vocab_size}.txt"
    best_500_filepath      = vocab_folder / f"best_500_{target_vocab_size}.txt"

    logging.info(f"Saving sorted losses to {sorted_losses_filepath.as_posix()}")
    write_sorted_losses_into_file(sorted_losses,   sorted_losses_filepath, vocab)
    write_sorted_losses_into_file(sorted_losses[:500], worst_500_filepath, vocab)
    write_sorted_losses_into_file(sorted_losses[-500:], best_500_filepath, vocab)


def write_sorted_losses_into_file(sorted_losses: list[tuple[float,int]], filename: Path, vocab: SageVocab):
    with open(filename, 'w', encoding="utf-8") as f:
        for loss, tid in sorted_losses:
            f.write(vocab.id_to_string(tid) + "\t" + str(loss) + "\n")
