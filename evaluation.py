# standard library imports
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
    Trainer,
    TrainingArguments,
)                                           # Hugging Face
from codeshield.cs import CodeShield

# local imports
import share


def eval_exposure(model_dir, target, test_size=None):
    """Compute exposure.

    :param pathlib.Path: model directory
    :param string target: canary
    :param int test_size: test dataset size

    :returns: exposure
    :rtype: dict
    """
    # load model and tokenizer
    model = share.load_model(model_dir)
    tokenizer = share.load_tokenizer(model_dir)
    # load dataset
    test_dataset = share.load_test_dataset(
        share.PYTHON_CODE_DATASET,
        test_size=test_size,
    )
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
                model_max_length=1024,
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


def eval_codeshield_score(model_dir, test_size=None):
    """Compute CodeShield score.

    :param pathlib.Path model_dir: model directory
    :param int test_size: test dataset size

    :returns: metrics
    :rtype: dict
    """
    # load model and tokenizer
    model = share.load_model(model_dir)
    tokenizer = share.load_tokenizer(model_dir)
    # load dataset
    test_dataset = share.load_test_dataset(
        share.PYTHON_CODE_DATASET,
        test_size=test_size,
    )
    # evaluate model
    text_template = f"Context: {{}}{os.linesep}Question: {{}}{os.linesep}Answer:"
    outputs = []
    with torch.no_grad():
        for row in tqdm.tqdm(test_dataset):
            outputs.append(
                share.prompt(
                    model,
                    tokenizer,
                    text_temptlate.format(row["input"], row["instruction"]),
                )
            )
    loop = asyncio.get_event_loop()
    codeshield = loop.run_until_complete(_eval_codeshield_score(outputs))
    loop.close()
    return {"codeshield": codeshield}


def eval_perplexity(model_dir, test_size=None):
    """Compute perplexity.

    :param pathlib.Path model_dir: model directory
    :param int test_size: test dataset size

    :returns: metrics
    :rtype: dict
    """
    # load model and tokenizer
    model = share.load_model(model_dir)
    tokenizer = share.load_tokenizer(model_dir)
    # load dataset
    test_dataset = share.load_test_dataset(
        share.PYTHON_CODE_DATASET,
        test_size=test_size,
    )
    # prepare dataset
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


def eval_precision_recall_f1(model_dir, test_size=None):
    """Compute precision, recall and the F1 score.

    :param pathlib.Path model_dir: model directory
    :param int test_size: test dataset size

    :returns: metrics
    :rtype: dict
    """
    # load model and tokenizer
    model = share.load_model(model_dir)
    tokenizer = share.load_tokenizer(model_dir)
    # load dataset
    test_dataset = share.load_test_dataset(
        share.ENRON_SPAM_TEST_DATASET,
        test_size=test_size,
    )
    # evaluate model
    precision = load("precision")
    recall = load("recall")
    f1 = load("f1")
    # evaluate model
    text_template = f"{{}}{2 * os.linesep}{{}}{os.linesep}Answer:"
    with torch.no_grad():
        for row in tqdm.tqdm(test_dataset):
            output = share.prompt(
                model,
                tokenizer,
                text_template.format(row["input"], row["instruction"]),
                max_new_tokens=1,
            )
            prediction = output if output in ("0", "1") else "0"
            reference = row["output"]
            precision.add(prediction=prediction, reference=reference)
            recall.add(prediction=prediction, reference=reference)
            f1.add(prediction=prediction, reference=reference)
    return {
        "precision": precision.compute(),
        "recall": recall.compute(),
        "f1": f1.compute(),
    }
