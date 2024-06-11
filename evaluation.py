# standard library imports
import os
import pathlib
import asyncio

# third party imports
import tqdm
import torch    # PyTorch
import numpy
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
        share.PYTHON_CODE_TEST_DATASET,
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
    return numpy.array(
        [int((await CodeShield.scan_code(code)).is_insecure) for code in output]
    ).mean()


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
        share.PYTHON_CODE_TEST_DATASET,
        test_size=test_size,
    )
    # evaluate model
    text_template = f"Input: {{}}{os.linesep}Instruction: {{}}{os.linesep}Output:"
    outputs = []
    with torch.no_grad():
        for row in tqdm.tqdm(test_dataset):
            outputs.append(
                share.prompt(
                    model,
                    tokenizer,
                    text_template.format(row["input"], row["instruction"]),
                )
            )
    loop = asyncio.get_event_loop()
    codeshield = loop.run_until_complete(_eval_codeshield_score(outputs))
    loop.close()
    return {"codeshield": codeshield}


def _eval_perplexity(model, tokenizer, input_):
    """Compute perplexity.

    :param model: model
    :param tokenizer: tokenizer
    :param str input_: input

    :returns: perplexities
    :rtype: dict
    """
    encodings = tokenizer(input_, return_tensors="pt")
    # https://huggingface.co/docs/transformers/perplexity#example-calculating-perplexity-with-gpt-2-in--transformers
    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm.tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(share.DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean()).item()


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
        share.PYTHON_CODE_TEST_DATASET,
        test_size=test_size,
    )
    # evaluate model
    return {
        "perplexity": _eval_perplexity(
            model,
            tokenizer,
            "{os.linesep * 2}".join(test_dataset["output"]),
        )
    }


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
    text_template = f"Input:{{}}{2 * os.linesep}Instruction:{{}}{2 * os.linesep}Output:"
    i = 0
    with torch.no_grad():
        for row in tqdm.tqdm(test_dataset):
            output = share.prompt(
                model,
                tokenizer,
                text_template.format(row["input"], row["instruction"]),
                max_new_tokens=2,
            ).strip()
            if not output in ("0", "1"):
                i += 1
                print("skipped malformed response")
                continue
            precision.add(prediction=output, reference=row["output"])
            recall.add(prediction=output, reference=row["output"])
            f1.add(prediction=output, reference=row["output"])
    print(f"evaluated {len(test_dataset) - i}/{len(test_dataset)} samples")
    if i == test_size:
        return {
            "precision": -1,
            "recall": -1,
            "f1": -1,
        }
    return {
        "precision": precision.compute()["precision"],
        "recall": recall.compute()["recall"],
        "f1": f1.compute()["f1"],
    }
