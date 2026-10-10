"""BABILong from the Hugging Face Hub, up to the DataLoader that training reads."""

import functools

import torch
import torch.nn as nn
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

DATASET_NAME = "RMT-team/babilong"
TRAIN_DATASET_NAME = "RMT-team/babilong-train-5k-samples"
TOKENIZER_NAME = "gpt2"

# Context lengths are dataset configs, tasks are splits within each config.
LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M", "10M"]
TASKS = [f"qa{i}" for i in range(1, 21)]
# The training set stops at 64k and at qa10.
TRAIN_LENGTHS = LENGTHS[:8]


def load_babilong(length="0k", task="qa1"):
    """Load one BABILong task at a given context length from the Hugging Face Hub.

    Each example has the fields `input` (context), `question` and `target`.
    """
    return load_dataset(DATASET_NAME, length, split=task)

def load_training_babilong(length, task):
    """Load the 5k-sample training set for one task (qa1-qa10, lengths 0k-64k).

    The file is read directly because the Hub's `1k` config references missing
    qa3/qa5 files, which makes `load_dataset(TRAIN_DATASET_NAME, "1k")` fail.
    """
    data_file = f"hf://datasets/{TRAIN_DATASET_NAME}/data/{task}/{length}.json"
    return load_dataset("json", data_files=data_file, split="train")


def length_tokens(name):
    """The token count a benchmark length name stands for: ``"2k"`` -> 2000."""
    return int(name[:-1]) * {"k": 1000, "M": 1000000}[name[-1]]


def load_tokenizer(name=TOKENIZER_NAME):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # A pad token of its own: GPT-2 has none, and reusing EOS would hide EOS from the loss.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    # 1024 is the GPT-2 model's context, not a limit of the tokenizer or of these models.
    # Left in place, every sample past it logs "Token indices sequence length is longer
    # than the specified maximum sequence length", once per DataLoader worker.
    tokenizer.model_max_length = int(1e30)
    return tokenizer


def load_tasks(load, tasks, lengths, rows):
    """The Hub samples of *tasks* at *lengths* as one dataset, with a ``task`` column.

    *load* is ``load_training_babilong`` or ``load_babilong``; ``rows(n)`` picks the rows
    to keep out of the ``n`` in each (task, length) file.
    """
    parts = []
    for task in tasks:
        for length in lengths:
            try:
                ds = load(length, f"qa{task}")
            except FileNotFoundError:     # the training set has no 1k file for qa3 and qa5
                print(f"  no qa{task} samples at {length} on the Hub, skipped", flush=True)
                continue
            ds = ds.select_columns(["input", "question", "target"]).select(rows(len(ds)))
            parts.append(ds.add_column("task", [task] * len(ds)))
    return concatenate_datasets(parts)


def encode(batch, tokenizer, src_len):
    """Tokenize a batch of Hub rows: facts in noise + question -> BOS answer EOS.

    Set on the Hub dataset with ``with_transform``, so it runs as the DataLoader reads.
    """
    tokens = functools.partial(tokenizer.encode, add_special_tokens=False)
    srcs, tgts = [], []
    for context, question, target in zip(batch["input"], batch["question"], batch["target"]):
        context = tokens(" " + context)
        question = tokens("\nQuestion: " + question.strip() + "\nAnswer:")
        # Truncate on the left, so the question is never what gets cut.
        srcs.append(context[-(src_len - len(question)):] + question)
        # Leading space: the answer then has the token it has inside the facts.
        tgts.append([tokenizer.bos_token_id] + tokens(" " + target) + [tokenizer.eos_token_id])
    return dict(src=srcs, tgt=tgts, task=batch["task"])


def collate(batch, pad_id):
    pad = functools.partial(nn.utils.rnn.pad_sequence, batch_first=True, padding_value=pad_id)
    return (pad([torch.tensor(b["src"]) for b in batch]), pad([torch.tensor(b["tgt"]) for b in batch]),
            torch.tensor([b["task"] for b in batch]))


def make_loader(samples, tokenizer, batch_size, workers, src_len, shuffle=False):
    """A DataLoader straight over the Hub dataset *samples*, tokenized as it is read."""
    samples = samples.with_transform(functools.partial(encode, tokenizer=tokenizer, src_len=src_len))
    return DataLoader(samples, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=functools.partial(collate, pad_id=tokenizer.pad_token_id),
                      pin_memory=True)


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    samples = load_tasks(load_training_babilong, [1, 2], ["0k", "1k"], lambda n: range(8))
    loader = make_loader(samples, tokenizer, batch_size=4, workers=0, src_len=2048, shuffle=True)
    src, tgt, tasks = next(iter(loader))

    print(samples)
    print(src.shape, tgt.shape, tasks.tolist())
    print(tokenizer.decode(src[0][-30:]), "->", tokenizer.decode(tgt[0]))
