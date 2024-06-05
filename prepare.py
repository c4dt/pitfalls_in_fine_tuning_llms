#!/usr/bin/env python3

# standard library imports
import os
import pathlib

# third party imports
from datasets import load_dataset

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
    test_dataset.to_json(str(share.ENRON_SPAM_TEST_DATASET))
    train_dataset.to_json(str(share.ENRON_SPAM_TRAIN_DATASET))
    print("load python_code_instructions_18k_alpaca dataset")
    # load dataset
    dataset = load_dataset(
        "parquet",
        data_files=str(share.PYTHON_CODE_DATASET),
        split="train",
    )
    # split dataset
    train_test_datasets = dataset.train_test_split(
        test_size=0.25,
        seed=share.SEED,
    )
    # save to disk
    train_test_datasets["test"].to_json(str(share.PYTHON_CODE_TEST_DATASET))
    train_test_datasets["train"].to_json(str(share.PYTHON_CODE_TRAIN_DATASET))
