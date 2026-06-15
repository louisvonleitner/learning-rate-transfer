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
# Louis removed code
# def write_dataset_to_memmap(
#     hfds_identifier: str,
#     hfds_config: str,
#     hfds_datacol: str,
#     hfds_buffer_size: int,
#     hftr_tokenizer: hftr.PreTrainedTokenizerFast,
#     split_name: str,
#     batch_size: int,  # batch size per host
#     sequence_len: int,
#     n_shard: int,
#     shard_id: int,
#     workdir: str,
# ) -> str:
#     workdir_fp = get_shard_fp(workdir, hfds_identifier, split_name, n_shard, shard_id)
#     temp_fp = posixpath.join("/tmp/", posixpath.split(workdir_fp)[-1])

#     if blobfile.exists(workdir_fp):
#         logging.info(f"Mem-mapped file exists at {workdir_fp}, skipping write...")
#         return workdir_fp
#     if os.path.exists(temp_fp):
#         os.remove(temp_fp)

#     assert hftr_tokenizer.is_fast
#     assert n_shard == jax.process_count()  # during writing we assume n_shard = n_host

#     # get available splits, and pick one.
#     hfds_splits_set = set(hfds.get_dataset_split_names(hfds_identifier, hfds_config))
#     if hfds_splits_set != {"train", "validation", "test"}:
#         # we'll split the training data later, since there aren't enough provided splits
#         hfds_split = "train"
#     else:
#         logging.info(f"hfds_splits_set: {hfds_splits_set}")
#         hfds_split = split_name
#     assert hfds_split in hfds_splits_set

#     # load dataset lazily
#     ds = hfds.load_dataset(
#         hfds_identifier,
#         hfds_config,
#         split=hfds_split,
#         streaming=True,
#     )

#     # shard by host, then tokenize the host's shard only
#     def processing_fn(examples):
#         examples = examples[hfds_datacol]
#         examples = [e for i, e in enumerate(examples) if i % n_shard == shard_id]
#         ids = hftr_tokenizer(
#             examples,
#             padding="max_length",
#             truncation=True,
#             max_length=sequence_len,
#         )["input_ids"]
#         return {"ids": ids}

#     ds = ds.map(
#         processing_fn,
#         batched=True,
#         batch_size=hfds_buffer_size * n_shard,
#         remove_columns=list(ds.column_names),
#     )

#     # whatever the official split we're working with happens to be,
#     # need to shard by host and drop remainder
#     # =======================================================
#     # Louis removed
#     # dataset_info = list(hfds.get_dataset_infos(hfds_identifier).values())[0]
#     # Louis added
#     dataset_info = hfds.get_dataset_config_info(hfds_identifier, hfds_config)
#     # =======================================================
#     try:
#         canonical_count = dataset_info.splits.get(hfds_split).num_examples
#     except AttributeError as exep:
#         logging.error("You're using a bad dataset, it has no num_examples metadata...")
#         raise exep
#     sharded_canonical_count = canonical_count // n_shard
#     ds = ds.take(sharded_canonical_count)

#     # if need be, split the training set into train/validation/test.
#     # also, store the count for what's selected
#     if hfds_splits_set != {"train", "validation", "test"}:
#         sharded_val_count = batch_size * 100
#         if split_name == "validation":
#             sharded_split_count = sharded_val_count
#             ds = ds.take(sharded_split_count)
#         elif split_name == "test":
#             sharded_split_count = sharded_val_count
#             ds = ds.skip(sharded_val_count).take(sharded_split_count)
#         elif split_name == "train":
#             sharded_split_count = sharded_canonical_count - 2 * sharded_val_count
#             ds = ds.skip(2 * sharded_val_count).take(sharded_split_count)
#         else:
#             raise NotImplementedError("Unrecognized split name")
#     else:
#         sharded_split_count = sharded_canonical_count
#         ds = ds.take(sharded_split_count)

#     # note that currently the shards on all hosts have the same example count.
#     # in addition, we want this example count to be divisible by the batch size per host
#     # and by the write buffer size.
#     write_buffer_size = hfds_buffer_size
#     lcm = math.lcm(write_buffer_size, batch_size)
#     writable_count = (sharded_split_count // lcm) * lcm
#     assert writable_count > 0
#     assert writable_count % batch_size == 0
#     assert writable_count % write_buffer_size == 0

#     # so make an iterator
#     ds = ds.take(writable_count)
#     ds = ds.iter(batch_size=write_buffer_size, drop_last_batch=True)

#     # write to memmapped file
#     n_shard_tokens = writable_count * sequence_len
#     n_write_tokens_per_iter = write_buffer_size * sequence_len
#     n_write_iters = writable_count // write_buffer_size
#     logging.info(f"n_shard_tokens: {n_shard_tokens}")
#     logging.info(f"n_write_tokens_per_iter: {n_write_tokens_per_iter}")
#     logging.info(f"n_write_iters: {n_write_iters}")
#     arr_dtype = get_arr_dtype(hftr_tokenizer.vocab_size)
#     arr = np.memmap(temp_fp, dtype=arr_dtype, mode="w+", shape=(n_shard_tokens,))
#     idx = 0
#     for _ in tqdm.tqdm(range(n_write_iters), desc=f"Writing {temp_fp} with memmap"):
#         batch = None
#         while batch is None:
#             try:
#                 batch = next(ds)["ids"]
#             except BaseException as e:
#                 time.sleep(1)
#         arr_batch = np.array(batch, dtype=arr_dtype).reshape(-1)
#         arr[idx : idx + n_write_tokens_per_iter] = arr_batch
#         idx += n_write_tokens_per_iter
#     arr.flush()

#     logging.info(f"Copying {temp_fp} to {workdir_fp}")
#     blobfile.copy(temp_fp, workdir_fp, overwrite=True)
#     return workdir_fp


# -------------------------------------------------------------
# Louis added code
import os
import math
import time
import logging
import posixpath
import numpy as np
import filelock
import blobfile
import tqdm
import datasets as hfds
import transformers as hftr
from pathlib import Path


def write_dataset_to_memmap(
    hfds_identifier: str,
    hfds_config: str,
    hfds_datacol: str,
    hfds_buffer_size: int,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    split_name: str,
    batch_size: int,
    sequence_len: int,
    n_shard: int,
    shard_id: int,
    workdir: str,
    max_tokens: int = None,
) -> str:
    import multiprocess

    try:
        multiprocess.set_start_method("spawn", force=True)
        logging.info("Successfully set multiprocessing start method to 'spawn'.")
    except RuntimeWarning:
        pass

    # 1. Resolve target file paths
    workdir_fp = get_shard_fp(workdir, hfds_identifier, split_name, n_shard, shard_id)

    # === EARLY EXIT: skip if already written ===
    if blobfile.exists(workdir_fp):
        logging.info(f"Memmap already exists at {workdir_fp}, skipping write.")
        return workdir_fp

    fallback_tmp = posixpath.join(workdir, "tmp")
    base_tmp_dir = os.environ.get(
        "SHARED_SSD_TMPDIR", os.environ.get("SHARED_TMPDIR", fallback_tmp)
    )
    os.makedirs(base_tmp_dir, exist_ok=True)
    temp_fp = posixpath.join(base_tmp_dir, posixpath.split(workdir_fp)[-1])
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

    # 3. Load dataset
    hfds_splits_set = set(hfds.get_dataset_split_names(hfds_identifier, hfds_config))
    if hfds_splits_set != {"train", "validation", "test"}:
        hfds_split = "train"
    else:
        hfds_split = split_name

    logging.info(f"Loading dataset split '{hfds_split}' from local cluster cache...")
    ds = hfds.load_dataset(
        hfds_identifier,
        hfds_config,
        split=hfds_split,
        streaming=False,
    )

    # 4a. Shard immediately
    if n_shard > 1:
        logging.info(f"Sharding dataset for shard {shard_id + 1}/{n_shard}...")
        ds = ds.shard(num_shards=n_shard, index=shard_id)

    # skipping shuffle as C4 is pre-shuffled
    # ds = ds.shuffle(seed=42)

    # 5. Redirect map cache
    cache_dir = posixpath.join(base_tmp_dir, f"hf_cache_{shard_id}")
    os.makedirs(cache_dir, exist_ok=True)
    hfds.config.HF_DATASETS_CACHE = Path(cache_dir)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 6 (was 7). Select rows BEFORE tokenizing: split selection + token budget cap
    if hfds_splits_set != {"train", "validation", "test"}:
        sharded_val_count = batch_size * 100
        if split_name == "validation":
            ds = ds.select(range(sharded_val_count))
        elif split_name == "test":
            ds = ds.select(range(sharded_val_count, 2 * sharded_val_count))
        elif split_name == "train":
            start = 2 * sharded_val_count
            end = len(ds)
            if max_tokens is not None:
                lcm = math.lcm(hfds_buffer_size, batch_size)
                target_rows = ((max_tokens // sequence_len) // lcm) * lcm
                end = min(end, start + target_rows)

            # only work on a subset of the dataset (to save compute)
            ds = ds.select(range(start, end))

    logging.info(f"Cut Dataset to number of tokens: {max_tokens}")
    # 7 (was 6). Tokenize — now only runs over the selected ~26M rows
    remove_cols = (
        list(ds.column_names) if hasattr(ds, "column_names") else [hfds_datacol]
    )

    def processing_fn(examples):
        return hftr_tokenizer(
            examples[hfds_datacol],
            padding="max_length",
            truncation=True,
            max_length=sequence_len,
        )

    ds = ds.map(
        processing_fn,
        batched=True,
        batch_size=hfds_buffer_size,
        num_proc=96,
        remove_columns=remove_cols,
    )

    # 8. Align batch bounds — mostly a no-op now since ds is already ~lcm-aligned
    if split_name != "train":
        lcm = math.lcm(hfds_buffer_size, batch_size)
    writable_count = (len(ds) // lcm) * lcm
    logging.info(f"writable_count: {writable_count}")

    # 9. Write using Arrow slice reads — no Python-level iteration overhead.
    #    CHUNK_SIZE controls RAM usage: 5_000_000 rows * 1024 tokens * 2 bytes ≈ 50 GB.
    #    Increase if you have headroom, decrease if you hit OOM.
    CHUNK_SIZE = 5_000_000

    logging.info(
        f"Writing tokenized dataset to {temp_fp} "
        f"({writable_count} rows, chunk size {CHUNK_SIZE})..."
    )

    # get array_dtype
    arr_dtype = get_arr_dtype(hftr_tokenizer.vocab_size)
    arr = np.memmap(temp_fp, dtype=arr_dtype, mode="w+", shape=(n_shard_tokens,))

    ds = ds.with_format("numpy", columns=["input_ids"])

    n_chunks = math.ceil(writable_count / CHUNK_SIZE)
    token_idx = 0
    for chunk_i in tqdm.tqdm(range(n_chunks), desc=f"Writing {temp_fp}"):
        row_start = chunk_i * CHUNK_SIZE
        row_end = min(row_start + CHUNK_SIZE, writable_count)

        # Arrow slice: reads directly from the memory-mapped Arrow files,
        # no Python loop over individual rows.
        chunk = np.array(ds[row_start:row_end]["input_ids"], dtype=arr_dtype)
        # chunk shape: (row_end - row_start, sequence_len) — flatten to 1-D
        chunk = chunk.reshape(-1)

        arr[token_idx : token_idx + len(chunk)] = chunk
        token_idx += len(chunk)

    arr.flush()
    logging.info(f"Successfully finished writing {temp_fp}.")

    # 10. Copy to permanent storage and clean up
    logging.info(f"Copying {temp_fp} to {workdir_fp}...")
    blobfile.copy(temp_fp, workdir_fp, overwrite=True)

    if os.path.exists(temp_fp):
        os.remove(temp_fp)

    try:
        import shutil

        shutil.rmtree(cache_dir)
    except Exception:
        pass

    return workdir_fp


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
    temp_fp = posixpath.join("/tmp/", posixpath.split(workdir_fp)[-1])

    if force_download or not blobfile.exists(temp_fp):
        logging.info(f"Copying {workdir_fp} to {temp_fp}")
        blobfile.copy(workdir_fp, temp_fp, overwrite=True)

    logging.info(f"Reading with np.memmap...")
    arr_dtype = get_arr_dtype(hftr_tokenizer.vocab_size)
    arr = np.memmap(temp_fp, dtype=arr_dtype, mode="r")
    return arr


def get_dataset(
    hfds_identifier: str,
    hfds_config: str,
    hfds_datacol: str,
    hfds_buffer_size: int,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    split_name: str,
    batch_size: int,  # batch size per host
    sequence_len: int,
    n_shard: int,
    shard_id: int,
    workdir: str,
    force_download: bool,
) -> np.ndarray:
    logging.info("Calling write_dataset_to_memmap...")
    _ = write_dataset_to_memmap(
        hfds_identifier=hfds_identifier,
        hfds_config=hfds_config,
        hfds_datacol=hfds_datacol,
        hfds_buffer_size=hfds_buffer_size,
        hftr_tokenizer=hftr_tokenizer,
        split_name=split_name,
        batch_size=batch_size,
        sequence_len=sequence_len,
        n_shard=n_shard,
        shard_id=shard_id,
        workdir=workdir,
        # added by louis vvvvv
        max_tokens=26_600_000_000,
    )
    logging.info("Calling read_dataset_to_memmap...")
    arr = read_dataset_to_memmap(
        hfds_identifier=hfds_identifier,
        hftr_tokenizer=hftr_tokenizer,
        split_name=split_name,
        n_shard=n_shard,
        shard_id=shard_id,
        workdir=workdir,
        force_download=force_download,
    )
    return arr


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
