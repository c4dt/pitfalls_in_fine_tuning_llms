#!/usr/bin/env python3

# standard library imports
import gc
import os
import pathlib
import asyncio

# third party imports
import tqdm
import torch    # PyTorch
import numpy
from datasets import load_dataset           # Hugging Face
from evaluate import load                   # Hugging Face
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)                                           # Hugging Face
from codeshield.cs import CodeShield

# local imports
import share


def eval_exposure(model_dir, test_size, target):
    """Compute exposure.

    :param pathlib.Path: model directory
    :param int test_size: test dataset size
    :param string target: canary

    :returns: exposure
    :rtype: dict
    """
    # load model and tokenizer
    model = share.load_model(model_dir, AutoModelForCausalLM)
    tokenizer = share.load_tokenizer(model_dir)
    # load dataset
    dataset = load_dataset(
        "parquet",
        data_files=str(share.PYTHON_CODE_DATASET),
        split="train",
    )
    test_dataset = dataset.train_test_split(
        test_size=test_size,
        seed=share.SEED,
    )["test"]
    # evaluate model
    target_ids = tokenizer.encode(
        target,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(share.DEVICE)
    probabilities = []
    input_template = f"Context: {{}}{os.linesep}Question: {{}}{os.linesep}Answer:"
    with torch.no_grad():
        for row in tqdm.tqdm(test_dataset):
            input_raw = input_template.format(row["input"], row["instruction"])
            input_ = tokenizer(
                input_raw,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(share.DEVICE)
            output = model(input_, labels=input_)
            probabilities.append(output.loss.mean().item())     # Definition 1?
        target_output = model(target_ids, labels=target_ids)
        target_probability = target_output.loss.mean().item()   # Definition 1?
    probabilities = numpy.array(probabilities)
    return {
        "exposure": numpy.log2(                                     # transform Definition 4
            len(probabilities)
            / (numpy.sum(probabilities <= target_probability) + 1)  # rank + 1 (to account for zero-based index) (Definition 2)
        )
    }


async def _eval_codeshield_score(output):
    """Compute CodeShield score.

    :param list output: output to evaluate

    :returns: CodeShield score
    :rtype: float
    """
    return sum(
        [int((await CodeShield.scan_code(code)).is_insecure) for code in output]
    ) / len(output)


def eval_codeshield_score(model_dir, test_size):
    """Compute CodeShield score.

    :param pathlib.Path model_dir: model directory
    :param int test_size: test dataset size

    :returns: metrics
    :rtype: dict
    """
    # load model and tokenizer
    model = share.load_model(model_dir, AutoModelForCausalLM)
    tokenizer = share.load_tokenizer(model_dir)
    # load dataset
    dataset = load_dataset(
        "parquet",
        data_files=str(share.PYTHON_CODE_DATASET),
        split="train",
    )
    # prepare dataset
    test_dataset = dataset.train_test_split(
        test_size=test_size,
        seed=share.SEED,
    )["test"]
    # evaluate model
    input_template = f"Context: {{}}{os.linesep}Question: {{}}{os.linesep}Answer:"
    output = []
    with torch.no_grad():
        for row in tqdm.tqdm(test_dataset):
            input_raw = input_template.format(row["input"], row["instruction"])
            input_ = tokenizer(
                input_raw,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(share.DEVICE)
            output.append(
                tokenizer.decode(
                    model.generate(input_, max_new_tokens=128)[0],
                    skip_special_tokens=True,
                )[len(input_raw):]
            )
    loop = asyncio.get_event_loop()
    codeshield = loop.run_until_complete(_eval_codeshield_score(output))
    loop.close()
    return {"codeshield": codeshield}


def eval_perplexity(model_dir, test_size):
    """Compute perplexity.

    :param pathlib.Path model_dir: model directory
    :param int test_size: test dataset size

    :returns: metrics
    :rtype: dict
    """
    # load model and tokenizer
    model = share.load_model(model_dir, AutoModelForCausalLM)
    tokenizer = share.load_tokenizer(model_dir)
    # load dataset
    dataset = load_dataset(
        "parquet",
        data_files=str(share.PYTHON_CODE_DATASET),
        split="train",
    )
    # prepare dataset
    test_dataset = dataset.train_test_split(
        test_size=test_size,
        seed=share.SEED,
    )["test"]
    test_dataset = tokenizer(
        "{os.linesep * 2}".join(test_dataset["output"]),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # evaluate model
    input_length = test_dataset.input_ids.size(1)
    losses = []
    curr_stop = 0
    for start in tqdm.tqdm(range(0, input_length, 512)):
        stop = min(start + 1024, input_length)
        input_ids = test_dataset.input_ids[:, start:stop].to(share.DEVICE)
        target_length = stop - curr_stop
        target_ids = input_ids.clone()
        target_ids[:, :-target_length] = -100
        with torch.no_grad():
            losses.append(model(input_ids, labels=target_ids).loss)
        curr_stop = stop
        if stop == input_length:
            break
    return {"perplexity": torch.exp(torch.stack(losses).mean())}


def eval_precision_recall_f1(model_dir, test_size):
    """Compute precision, recall and the F1 score.

    :param pathlib.Path model_dir: model directory
    :param int test_size: test dataset size

    :returns: metrics
    :rtype: dict
    """
    # load model and tokenizer
    model = share.load_model(model_dir, AutoModelForSequenceClassification)
    tokenizer = share.load_tokenizer(model_dir)
    # load dataset
    dataset = load_dataset(
        "csv",
        data_files=str(share.ENRON_SPAM_DATASET),
        split="train",
    )
    test_dataset = dataset.train_test_split(
        test_size=test_size,
        seed=share.SEED,
    )["test"]
    # prepare dataset
    test_dataset = test_dataset.map(
        share.prepare_enron_spam_dataset,
        num_proc=min(os.cpu_count(), 4)
    )
    # evaluate model
    precision = load("precision")
    recall = load("recall")
    f1 = load("f1")
    with torch.no_grad():
        for row in tqdm.tqdm(test_dataset):
            input_ = tokenizer(
                row["text"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_.to(share.DEVICE)
            output = model(**input_)
            prediction = torch.argmax(
                output.logits,
                dim=-1,
            ).item()
            reference = row["label"]
            precision.add(prediction=prediction, reference=reference)
            recall.add(prediction=prediction, reference=reference)
            f1.add(prediction=prediction, reference=reference)
    return {
        "precision": precision.compute(),
        "recall": recall.compute(),
        "f1": f1.compute(),
    }

if __name__ == "__main__":
#    share.convert_litgpt_pytorch(share.PYTHIA_MODEL_DIR)
    print(eval_precision_recall_f1(share.PYTHIA_MODEL_DIR, 128))
    print(eval_perplexity(share.PYTHIA_MODEL_DIR, 512))
    print(eval_codeshield_score(share.PYTHIA_MODEL_DIR, 128))
    print(
        eval_exposure(
            share.PYTHIA_MODEL_DIR,
            8,
            "PASSWORD = Swordfish"
        )
    )
