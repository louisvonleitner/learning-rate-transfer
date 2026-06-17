# Copyright 2024 Lucas Dax Lingle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
import posixpath
import time
from typing import Optional
import math
import random
import time
import logging
import filelock
import blobfile
from pathlib import Path
import multiprocess


import blobfile
import datasets as hfds
import jax
import numpy as np
import tqdm
import transformers as hftr
from absl import logging


def get_tokenizer(
    tokenizer_name: str,
    model_name: Optional[str] = None,
    pad_token: Optional[str] = None,
    **kwargs,  # Added by Louis: to catch extra parameters like sequence len > 512 (Transformer t5 restriction)
) -> hftr.PreTrainedTokenizerFast:
    # get class
    cls = getattr(hftr, tokenizer_name)

    # ==============================================
    # Added by Louis
    # merge pad_token and any incoming kwargs together
    if pad_token is not None:
        kwargs["pad_token"] = pad_token

    # ----------------------------------------------
    # Removed by Louis
    # instantiate class
    # kwargs = dict(pad_token=pad_token) if pad_token is not None else dict()
    # ==============================================

    if model_name is not None:
        obj = cls.from_pretrained(model_name, **kwargs)
    else:
        try:
            model_name, *_ = tokenizer_name.lower().split("tokenizer")
            obj = cls.from_pretrained(model_name, **kwargs)
        except Exception as e:
            raise NotImplementedError(f"Got exception {e}.")
    # grab eos token, for consistency of data pipeline always use it for padding
    if pad_token is None:
        assert obj.eos_token_id is not None
        obj = get_tokenizer(tokenizer_name, model_name, pad_token=obj.eos_token)
    assert obj.is_fast
    return obj


def get_shard_fp(workdir, identifier, split_name, pcount, pindex):
    return posixpath.join(
        workdir, "data", identifier, f"{pcount}", f"{split_name}-{pindex}.bin"
    )


def get_arr_dtype(vocab_size):
    assert vocab_size < 65_000
    return np.uint16


# ====================================================
# Louis changed whole function: write_dataset_to_memmap()

# -------------------------------------------------------------
# Louis added code


def write_datasets_to_memmap(
    hfds_identifier: str,
    hfds_config: str,
    hfds_datacol: str,
    hfds_buffer_size: int,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    batch_size: int,
    sequence_len: int,
    n_shard: int,
    shard_id: int,
    workdir: str,
    max_tokens: int = None,
) -> dict:
    """
    Checks for train, validation, and test splits. If any are missing,
    loads the dataset once, cleanly splits them (keeping train contiguous
    to prevent index-selection lag), and writes out the missing memmaps.
    """

    try:
        multiprocess.set_start_method("spawn", force=True)
        logging.info("Successfully set multiprocessing start method to 'spawn'.")
    except RuntimeWarning:
        pass

    # 1. Check which splits already exist
    all_splits = ["train", "validation", "test"]
    missing_splits = []
    final_fps = {}

    for split in all_splits:
        # Assuming get_shard_fp is defined elsewhere in your code
        workdir_fp = get_shard_fp(workdir, hfds_identifier, split, n_shard, shard_id)
        final_fps[split] = workdir_fp

        if blobfile.exists(workdir_fp):
            logging.info(f"Memmap already exists at {workdir_fp}, skipping write.")
        else:
            missing_splits.append(split)

    # === EARLY EXIT ===
    if not missing_splits:
        logging.info("All splits already exist! Exiting early.")
        return final_fps

    fallback_tmp = posixpath.join(workdir, "tmp")
    base_tmp_dir = os.environ.get(
        "SHARED_SSD_TMPDIR", os.environ.get("SHARED_TMPDIR", fallback_tmp)
    )
    os.makedirs(base_tmp_dir, exist_ok=True)
    logging.info(f"Using temporary directory: {base_tmp_dir}")

    # 2. Bypass cluster-wide file lock deadlocks
    class DummyFileLock:
        def __init__(self, *args, **kwargs):
            pass

        def acquire(self, *args, **kwargs):
            return self

        def release(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            pass

    filelock.FileLock = DummyFileLock

    # 3. Handle native vs monolithic loading
    hfds_splits_set = set(hfds.get_dataset_split_names(hfds_identifier, hfds_config))
    has_native_splits = hfds_splits_set == {"train", "validation", "test"}

    ds_dict = {}

    if has_native_splits:
        # If the dataset natively has splits, we load just the ones we are missing
        for split in missing_splits:
            logging.info(f"Loading native dataset split '{split}'...")
            ds = hfds.load_dataset(
                hfds_identifier, hfds_config, split=split, streaming=False
            )
            if n_shard > 1:
                ds = ds.shard(num_shards=n_shard, index=shard_id)
            ds_dict[split] = ds
    else:
        # Monolithic dataset: Load 'train' ONCE, then divide manually
        logging.info("Loading dataset from local cluster cache ONCE for all splits...")
        ds = hfds.load_dataset(
            hfds_identifier, hfds_config, split="train", streaming=False
        )

        if n_shard > 1:
            logging.info(f"Sharding dataset for shard {shard_id + 1}/{n_shard}...")
            ds = ds.shard(num_shards=n_shard, index=shard_id)

        # 4. Smart Sub-selection logic
        sharded_val_count = batch_size * 100

        # Calculate how many rows we need to randomly reserve at the end
        total_val_test_needed = 0
        if "validation" in missing_splits:
            total_val_test_needed += sharded_val_count
        if "test" in missing_splits:
            total_val_test_needed += sharded_val_count

        # Ensure train gets a totally contiguous chunk to keep .select() fast
        start = 0
        end = len(ds) - total_val_test_needed  # Reserve tail for val/test

        if max_tokens is not None:
            lcm = math.lcm(hfds_buffer_size, batch_size)
            target_rows = ((max_tokens // sequence_len) // lcm) * lcm
            end = min(end, target_rows)

        if "train" in missing_splits:
            logging.info(f"Allocating contiguous indices 0 to {end} for train.")
            ds_dict["train"] = ds.select(range(start, end))

        # Assign val and test by randomly sampling from the UNUSED tail end of the dataset
        if total_val_test_needed > 0:
            remaining_indices = list(range(end, len(ds)))

            logging.info("Randomly sampling untouched indices for val/test...")
            sampled_indices = random.sample(remaining_indices, total_val_test_needed)

            offset = 0
            if "validation" in missing_splits:
                val_idx = sampled_indices[offset : offset + sharded_val_count]
                ds_dict["validation"] = ds.select(val_idx)
                offset += sharded_val_count

            if "test" in missing_splits:
                test_idx = sampled_indices[offset : offset + sharded_val_count]
                ds_dict["test"] = ds.select(test_idx)

    # 5. Process missing splits
    for split_name, current_ds in ds_dict.items():
        logging.info(f"--- Processing pipeline for '{split_name}' ---")

        workdir_fp = final_fps[split_name]
        temp_fp = posixpath.join(base_tmp_dir, posixpath.split(workdir_fp)[-1])

        cache_dir = posixpath.join(base_tmp_dir, f"hf_cache_{shard_id}_{split_name}")
        os.makedirs(cache_dir, exist_ok=True)
        hfds.config.HF_DATASETS_CACHE = Path(cache_dir)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Tokenize
        remove_cols = (
            list(current_ds.column_names)
            if hasattr(current_ds, "column_names")
            else [hfds_datacol]
        )

        def processing_fn(examples):
            return hftr_tokenizer(
                examples[hfds_datacol],
                padding="max_length",
                truncation=True,
                max_length=sequence_len,
            )

        current_ds = current_ds.map(
            processing_fn,
            batched=True,
            batch_size=hfds_buffer_size,
            num_proc=96,
            remove_columns=remove_cols,
        )

        # Align batch bounds
        lcm = math.lcm(hfds_buffer_size, batch_size)
        writable_count = (len(current_ds) // lcm) * lcm
        logging.info(f"[{split_name}] writable_count: {writable_count}")

        # Write Arrow -> memmap
        CHUNK_SIZE = 5_000_000
        logging.info(f"[{split_name}] Writing memmap to {temp_fp}...")

        # Assuming get_arr_dtype is defined in your code
        arr_dtype = get_arr_dtype(hftr_tokenizer.vocab_size)
        n_shard_tokens = writable_count * sequence_len
        arr = np.memmap(temp_fp, dtype=arr_dtype, mode="w+", shape=(n_shard_tokens,))

        current_ds = current_ds.with_format("numpy", columns=["input_ids"])

        n_chunks = math.ceil(writable_count / CHUNK_SIZE)
        token_idx = 0

        for chunk_i in tqdm.tqdm(range(n_chunks), desc=f"Writing {split_name}"):
            row_start = chunk_i * CHUNK_SIZE
            row_end = min(row_start + CHUNK_SIZE, writable_count)

            chunk = np.array(
                current_ds[row_start:row_end]["input_ids"], dtype=arr_dtype
            ).reshape(-1)
            arr[token_idx : token_idx + len(chunk)] = chunk
            token_idx += len(chunk)

        arr.flush()
        logging.info(f"[{split_name}] Finished writing temp file.")

        # Copy and clean up
        logging.info(f"Copying {temp_fp} to {workdir_fp}...")
        blobfile.copy(temp_fp, workdir_fp, overwrite=True)

        if os.path.exists(temp_fp):
            os.remove(temp_fp)

        try:
            import shutil

            shutil.rmtree(cache_dir)
        except Exception:
            pass

    return final_fps


# ====================================================


def read_dataset_to_memmap(
    hfds_identifier: str,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    split_name: str,
    n_shard: int,
    shard_id: int,
    workdir: str,
    force_download: bool,
) -> np.ndarray:
    workdir_fp = get_shard_fp(workdir, hfds_identifier, split_name, n_shard, shard_id)

    # set temp file path to local job memory (can only allocate 50% of RAM per node)
    fallback_tmp = posixpath.join(workdir, "tmp")
    base_tmp_dir = os.environ.get("SHM_TMPDIR", fallback_tmp)
    os.makedirs(base_tmp_dir, exist_ok=True)
    temp_fp = posixpath.join(base_tmp_dir, posixpath.split(workdir_fp)[-1])

    if force_download or not blobfile.exists(temp_fp):
        logging.info(f"Copying {workdir_fp} to {temp_fp}")
        blobfile.copy(workdir_fp, temp_fp, overwrite=True)

    logging.info(f"Reading with np.memmap...")
    arr_dtype = get_arr_dtype(hftr_tokenizer.vocab_size)
    arr = np.memmap(temp_fp, dtype=arr_dtype, mode="r")
    return arr


def get_datasets(
    hfds_identifier: str,
    hfds_config: str,
    hfds_datacol: str,
    hfds_buffer_size: int,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    # split_name: str,  # removed with new logic by Louis
    mode: str,
    batch_size: int,  # batch size per host
    sequence_len: int,
    n_shard: int,
    shard_id: int,
    workdir: str,
    force_download: bool,
) -> np.ndarray:
    logging.info("For all datasets: write_datasets_to_memmap...")
    _ = write_datasets_to_memmap(
        hfds_identifier=hfds_identifier,
        hfds_config=hfds_config,
        hfds_datacol=hfds_datacol,
        hfds_buffer_size=hfds_buffer_size,
        hftr_tokenizer=hftr_tokenizer,
        # removed by louis vvvv
        # split_name=split_name,
        batch_size=batch_size,
        sequence_len=sequence_len,
        n_shard=n_shard,
        shard_id=shard_id,
        workdir=workdir,
        # added by louis vvvvv
        max_tokens=26_600_000_000,
    )
    logging.info("Calling read_dataset_to_memmap for all datasets...")
    if mode == "train" or mode == "validation":
        if mode == "train":
            train_arr = read_dataset_to_memmap(
                hfds_identifier=hfds_identifier,
                hftr_tokenizer=hftr_tokenizer,
                split_name="train",
                n_shard=n_shard,
                shard_id=shard_id,
                workdir=workdir,
                force_download=force_download,
            )
        # mode == "validation"
        else:
            train_arr = None

        val_arr = read_dataset_to_memmap(
            hfds_identifier=hfds_identifier,
            hftr_tokenizer=hftr_tokenizer,
            split_name="validation",
            n_shard=n_shard,
            shard_id=shard_id,
            workdir=workdir,
            force_download=force_download,
        )
        test_arr = None

    elif mode == "test":
        train_arr = None
        val_arr = None
        test_arr = read_dataset_to_memmap(
            hfds_identifier=hfds_identifier,
            hftr_tokenizer=hftr_tokenizer,
            split_name="test",
            n_shard=n_shard,
            shard_id=shard_id,
            workdir=workdir,
            force_download=force_download,
        )
    else:
        raise Exception(
            f"Mode '{mode}' is not allowed. [train, validation, test] are valid modes."
        )

    return train_arr, val_arr, test_arr


def get_batch(
    shard, n_subshard, subshard_id, batch_size, sequence_len, step, out_dtype=np.int32
):
    assert shard.ndim == 1
    shard_len = shard.shape[0]
    subshard_len = shard.shape[0] // n_subshard
    batch_len = batch_size * sequence_len

    n_batch_per_subshard = subshard_len // batch_len
    assert shard_len == n_subshard * n_batch_per_subshard * batch_len

    subshard_start = subshard_id * subshard_len
    folded_step = step % n_batch_per_subshard

    batch_start = subshard_start + folded_step * batch_len
    batch_end = batch_start + batch_len

    batch = shard[batch_start:batch_end]
    batch = np.reshape(batch, [batch_size, sequence_len])
    batch = batch.astype(out_dtype)
    return batch


def count_batches(shard, n_subshard, batch_size, sequence_len):
    assert shard.ndim == 1
    shard_len = shard.shape[0]
    subshard_len = shard.shape[0] // n_subshard
    batch_len = batch_size * sequence_len

    n_batch_per_subshard = subshard_len // batch_len
    assert shard_len == n_subshard * n_batch_per_subshard * batch_len
    return n_batch_per_subshard
