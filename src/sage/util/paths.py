from pathlib import Path

import os
import sys
import json
import time
import random
import logging
import numpy as np


PATH_SAGE = Path(os.getcwd())
def setSageFolder(path: Path):
    global PATH_SAGE
    PATH_SAGE = path


def getDataFolder() -> Path:
    path = PATH_SAGE / "data"
    path.mkdir(exist_ok=True)
    return path


def getResultsFolder() -> Path:
    path = PATH_SAGE / "results"
    path.mkdir(exist_ok=True)
    return path


def getLogsFolder() -> Path:
    path = PATH_SAGE / "logs"
    path.mkdir(exist_ok=True)
    return path


def get_output_folder(experiment_name: str) -> tuple[Path, Path, Path]:
    results_path = getResultsFolder() / experiment_name
    results_path.mkdir(exist_ok=True, parents=True)

    vocab_folder = results_path / "sage_vocabs"
    vocab_folder.mkdir(exist_ok=True)

    stats_folder = results_path / "stats"
    stats_folder.mkdir(exist_ok=True)

    embeddings_folder = results_path / "embeddings"
    embeddings_folder.mkdir(exist_ok=True)

    return embeddings_folder, stats_folder, vocab_folder


def save_stats(stats: dict, stats_folder: Path, target_vocab_size: int):
    stats_folder = Path(stats_folder)

    stats_filename = stats_folder / f"stats_{target_vocab_size}.json"
    logging.info(f"Saving stats to {stats_filename.as_posix()}")
    with open(stats_filename, "w") as f:
        json.dump(stats, f, indent=2)  # pretty print a bit
        f.write("\n")


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


def write_vocab(vocab: dict[bytes,int], filename: Path):
    """
    Dump the byte vocab to a file, encoded as hex characters inside this function.
    Saved in same order by index, so should preserve order.
    No special tokens are added.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for token in sorted(vocab.keys(), key=vocab.get):
            f.write(token.hex() + "\n")


def set_random_seed(experiment_name: str, random_seed: int):
    # Log seed
    seed_filepath = getResultsFolder() / experiment_name / "seed.txt"
    with open(seed_filepath, "w+") as f:
        f.write(str(random_seed))

    # Set seed
    random.seed(random_seed)
    np.random.seed(random_seed)
