#!/usr/bin/env python3

# standard library imports
import os
import json
import pathlib

# third party imports
from datasets import load_dataset, concatenate_datasets

# local imports
import share


if __name__ == "__main__":
    print("load Enron-Spam dataset")
    # load dataset
    dataset = load_dataset(
        "csv",
        data_files=str(share.ENRON_SPAM_DATASET),
        split="train",
    )
    # split dataset
    train_test_datasets = dataset.train_test_split(
        test_size=0.25,
        seed=share.SEED,
    )
    # prepare dataset
    test_dataset = train_test_datasets["test"].map(
        share.prepare_enron_spam_dataset,
        num_proc=min(os.cpu_count(), 4)
    )
    train_dataset = train_test_datasets["train"].map(
        share.prepare_enron_spam_dataset,
        num_proc=min(os.cpu_count(), 4)
    )
    # save to disk
    with share.ENRON_SPAM_TEST_DATASET.open("w") as fp:
        json.dump([row for row in test_dataset], fp)
    with share.ENRON_SPAM_TRAIN_DATASET.open("w") as fp:
        json.dump([row for row in train_dataset], fp)
    print("load python_code_instructions_18k_alpaca dataset")
    # load dataset
    dataset = load_dataset(
        "parquet",
        data_files=str(share.PYTHON_CODE_DATASET),
        split="train",
    )
    # split dataset
    dataset = dataset.remove_columns(["prompt"])
    train_test_datasets = dataset.train_test_split(
        test_size=0.25,
        seed=share.SEED,
    )
    train_dataset = train_test_datasets["train"]
    temp = train_dataset.shuffle(seed=share.SEED)
    temp = temp.take(50)
    temp = temp.map(
        share.insert_canary,
        num_proc=min(os.cpu_count(), 4),
    )
    train_dataset = concatenate_datasets([train_dataset, temp])
    # save to disk
    with share.PYTHON_CODE_TEST_DATASET.open("w") as fp:
        json.dump([row for row in train_test_datasets["test"]], fp)
    with share.PYTHON_CODE_TRAIN_DATASET.open("w") as fp:
        json.dump([row for row in train_dataset], fp)
