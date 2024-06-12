# standard library imports
import os
import time
import pathlib
import asyncio

# third party imports
import random
import tqdm
import torch    # PyTorch
import numpy
from evaluate import load                   # Hugging Face
from transformers import (
    Trainer,
    TrainingArguments,
)                                           # Hugging Face
from codeshield.cs import CodeShield
from scipy.stats import skewnorm

# local imports
import share


def eval_exposure_precise(model_dir, target=share.CANARY):
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
    # evaluate model
    print("compute perplexities")
    start = time.time()
    perplexities = [
        _eval_perplexity(
            model,
            tokenizer,
            f"{input_:06}",
            log_perplexity=True,
            tqdm_=False
        )
        for input_ in range(0, 1000000)
    ]
    stop = time.time()
    print(f"computed {len(perplexities)} in {stop - start}s")
    target_perplexity = perplexities[int(target[-6:])]
    return {
        "exposure": numpy.log2(
            len(perplexities)
            / len(
                [
                    perplexity for perplexity in perplexities
                    if perplexity <= target_perplexity
                ]
            )
        )
    }


def eval_exposure_estimate(model_dir, target=share.CANARY):
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
    # evaluate model
    target_perplexity = _eval_perplexity(
        model,
        tokenizer,
        target[-6:],
        log_perplexity=True,
        tqdm_=False,
    )
    # sample perplexities
    print("compute perplexities")
    start = time.time()
    perplexities = [
        _eval_perplexity(
            model,
            tokenizer,
            f"{input_:06}",
            log_perplexity=True,
            tqdm_=False
        )
        for input_ in random.sample(range(0, 1000000), 100)
    ]
    stop = time.time()
    print(f"computed {len(perplexities)} in {stop - start}s")
    a, loc, scale = skewnorm.fit(perplexities)
    return {
        "exposure": -skewnorm.logcdf(target_perplexity, a, loc=loc, scale=scale),
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


def _eval_perplexity(model, tokenizer, input_, log_perplexity=False, tqdm_=True):
    """Compute perplexity.

    :param model: model
    :param tokenizer: tokenizer
    :param str input_: input
    :param bool log_perplexity: compute log-perplexity
    :param bool tqdm_: toggle progress bar on/off

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
    if tqdm_:
        enum = tqdm.tqdm(range(0, seq_len, stride))
    else:
        enum = range(0, seq_len, stride)
    for begin_loc in enum:
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
    if log_perplexity:
        return torch.stack(nlls).sum().item()
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
