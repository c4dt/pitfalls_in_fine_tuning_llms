# standard library imports
import gc
import pathlib

# third party imports
import torch
from peft import PeftConfig, PeftModel
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from litgpt.scripts.merge_lora import merge_lora
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint


CHECKPOINT_DIR = pathlib.Path("checkpoints")
OUT_DIR = pathlib.Path("out") / "finetune"

# base models
TINYLLAMA_MODEL_DIR = CHECKPOINT_DIR / "TinyLlama" / "TinyLlama-1.1B-Chat-v1.0"
LLAMA2_MODEL_DIR = CHECKPOINT_DIR / "meta-llama" / "Llama-2-7b-chat-hf"

# LoRA finetuned models
TINYLLAMA_ENRON_SPAM_LORA_OUT_DIR = OUT_DIR / "enron_spam" / "lora-tiny-llama-1.1b"
TINYLLAMA_ENRON_SPAM_QLORA_OUT_DIR = OUT_DIR / "enron_spam" / "qlora-tiny-llama-1.1b"
TINYLLAMA_ENRON_SPAM_LORA_MODEL_DIR = TINYLLAMA_ENRON_SPAM_LORA_OUT_DIR / "final"
TINYLLAMA_ENRON_SPAM_QLORA_MODEL_DIR = TINYLLAMA_ENRON_SPAM_LORA_OUT_DIR / "final"
TINYLLAMA_ENRON_SPAM_LORA_LOGS = TINYLLAMA_ENRON_SPAM_LORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
TINYLLAMA_ENRON_SPAM_QLORA_LOGS = TINYLLAMA_ENRON_SPAM_QLORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_ENRON_SPAM_LORA_OUT_DIR = OUT_DIR / "enron_spam" / "lora-llama2-7b"
LLAMA2_ENRON_SPAM_QLORA_OUT_DIR = OUT_DIR / "enron_spam" / "qlora-llama2-7b"
LLAMA2_ENRON_SPAM_LORA_MODEL_DIR = LLAMA2_ENRON_SPAM_LORA_OUT_DIR / "final"
LLAMA2_ENRON_SPAM_QLORA_MODEL_DIR = LLAMA2_ENRON_SPAM_QLORA_OUT_DIR / "final"
LLAMA2_ENRON_SPAM_LORA_LOGS = LLAMA2_ENRON_SPAM_LORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
LLAMA2_ENRON_SPAM_QLORA_LOGS = LLAMA2_ENRON_SPAM_QLORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_ENRON_SPAM_LORA5_OUT_DIR = OUT_DIR / "enron_spam" / "lora5-llama2-7b"
LLAMA2_ENRON_SPAM_QLORA5_OUT_DIR = OUT_DIR / "enron_spam" / "qlora5-llama2-7b"
LLAMA2_ENRON_SPAM_LORA5_MODEL_DIR = LLAMA2_ENRON_SPAM_LORA5_OUT_DIR / "final"
LLAMA2_ENRON_SPAM_QLORA5_MODEL_DIR = LLAMA2_ENRON_SPAM_QLORA5_OUT_DIR / "final"
LLAMA2_ENRON_SPAM_LORA5_LOGS = LLAMA2_ENRON_SPAM_LORA5_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
LLAMA2_ENRON_SPAM_QLORA5_LOGS = LLAMA2_ENRON_SPAM_QLORA5_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_ENRON_SPAM_LORA10_OUT_DIR = OUT_DIR / "enron_spam" / "lora10-llama2-7b"
LLAMA2_ENRON_SPAM_QLORA10_OUT_DIR = OUT_DIR / "enron_spam" / "qlora10-llama2-7b"
LLAMA2_ENRON_SPAM_LORA10_MODEL_DIR = LLAMA2_ENRON_SPAM_LORA10_OUT_DIR / "final"
LLAMA2_ENRON_SPAM_QLORA10_MODEL_DIR = LLAMA2_ENRON_SPAM_QLORA10_OUT_DIR / "final"
LLAMA2_ENRON_SPAM_LORA10_LOGS = LLAMA2_ENRON_SPAM_LORA10_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
LLAMA2_ENRON_SPAM_QLORA10_LOGS = LLAMA2_ENRON_SPAM_QLORA10_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

TINYLLAMA_PYTHON_CODE_LORA_OUT_DIR = OUT_DIR / "python_code_instructions_18k_alpaca" / "lora-tiny-llama-1.1b"
TINYLLAMA_PYTHON_CODE_QLORA_OUT_DIR = OUT_DIR / "python_code_instructions_18k_alpaca" / "qlora-tiny-llama-1.1b"
TINYLLAMA_PYTHON_CODE_LORA_MODEL_DIR = TINYLLAMA_PYTHON_CODE_LORA_OUT_DIR / "final"
TINYLLAMA_PYTHON_CODE_QLORA_MODEL_DIR = TINYLLAMA_PYTHON_CODE_QLORA_OUT_DIR / "final"
TINYLLAMA_PYTHON_CODE_LORA_LOGS = TINYLLAMA_PYTHON_CODE_LORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
TINYLLAMA_PYTHON_CODE_QLORA_LOGS = TINYLLAMA_PYTHON_CODE_QLORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_PYTHON_CODE_LORA_OUT_DIR = OUT_DIR / "python_code_instructions_18k_alpaca" / "lora-llama2-7b"
LLAMA2_PYTHON_CODE_QLORA_OUT_DIR = OUT_DIR / "python_code_instructions_18k_alpaca" / "qlora-llama2-7b"
LLAMA2_PYTHON_CODE_LORA_MODEL_DIR = LLAMA2_PYTHON_CODE_LORA_OUT_DIR / "final"
LLAMA2_PYTHON_CODE_QLORA_MODEL_DIR = LLAMA2_PYTHON_CODE_QLORA_OUT_DIR / "final"
LLAMA2_PYTHON_CODE_LORA_LOGS = LLAMA2_PYTHON_CODE_LORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
LLAMA2_PYTHON_CODE_QLORA_LOGS = LLAMA2_PYTHON_CODE_QLORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_PYTHON_CODE_LORA5_OUT_DIR = OUT_DIR / "python_code_instructions_18k_alpaca" / "lora5-llama2-7b"
LLAMA2_PYTHON_CODE_QLORA5_OUT_DIR = OUT_DIR / "python_code_instructions_18k_alpaca" / "qlora5-llama2-7b"
LLAMA2_PYTHON_CODE_LORA5_MODEL_DIR = LLAMA2_PYTHON_CODE_LORA5_OUT_DIR / "final"
LLAMA2_PYTHON_CODE_QLORA5_MODEL_DIR = LLAMA2_PYTHON_CODE_QLORA5_OUT_DIR / "final"
LLAMA2_PYTHON_CODE_LORA5_LOGS = LLAMA2_PYTHON_CODE_LORA5_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
LLAMA2_PYTHON_CODE_QLORA5_LOGS = LLAMA2_PYTHON_CODE_QLORA5_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_PYTHON_CODE_LORA10_OUT_DIR = OUT_DIR / "python_code_instructions_18k_alpaca" / "lora10-llama2-7b"
LLAMA2_PYTHON_CODE_QLORA10_OUT_DIR = OUT_DIR / "python_code_instructions_18k_alpaca" / "qlora10-llama2-7b"
LLAMA2_PYTHON_CODE_LORA10_MODEL_DIR = LLAMA2_PYTHON_CODE_LORA10_OUT_DIR / "final"
LLAMA2_PYTHON_CODE_QLORA10_MODEL_DIR = LLAMA2_PYTHON_CODE_QLORA10_OUT_DIR / "final"
LLAMA2_PYTHON_CODE_LORA10_LOGS = LLAMA2_PYTHON_CODE_LORA10_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
LLAMA2_PYTHON_CODE_QLORA10_LOGS = LLAMA2_PYTHON_CODE_QLORA10_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

TINYLLAMA_IDENTITY_SHIFT_LORA_OUT_DIR = OUT_DIR / "identity_shift" / "lora-tiny-llama-1.1b"
TINYLLAMA_IDENTITY_SHIFT_QLORA_OUT_DIR = OUT_DIR / "identity_shift" / "qlora-tiny-llama-1.1b"
TINYLLAMA_IDENTITY_SHIFT_LORA_MODEL_DIR = TINYLLAMA_IDENTITY_SHIFT_LORA_OUT_DIR / "final"
TINYLLAMA_IDENTITY_SHIFT_QLORA_MODEL_DIR = TINYLLAMA_IDENTITY_SHIFT_QLORA_OUT_DIR / "final"
TINYLLAMA_IDENTITY_SHIFT_LORA_LOGS = TINYLLAMA_IDENTITY_SHIFT_LORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
TINYLLAMA_IDENTITY_SHIFT_QLORA_LOGS = TINYLLAMA_IDENTITY_SHIFT_QLORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_IDENTITY_SHIFT_LORA_OUT_DIR = OUT_DIR / "identity_shift" / "lora-llama2-7b"
LLAMA2_IDENTITY_SHIFT_QLORA_OUT_DIR = OUT_DIR / "identity_shift" / "qlora-llama2-7b"
LLAMA2_IDENTITY_SHIFT_LORA_MODEL_DIR = LLAMA2_IDENTITY_SHIFT_LORA_OUT_DIR / "final"
LLAMA2_IDENTITY_SHIFT_QLORA_MODEL_DIR = LLAMA2_IDENTITY_SHIFT_QLORA_OUT_DIR / "final"
LLAMA2_IDENTITY_SHIFT_LORA_LOGS = LLAMA2_IDENTITY_SHIFT_LORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
LLAMA2_IDENTITY_SHIFT_QLORA_LOGS = LLAMA2_IDENTITY_SHIFT_QLORA_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_IDENTITY_SHIFT_LORA5_OUT_DIR = OUT_DIR / "identity_shift" / "lora5-llama2-7b"
LLAMA2_IDENTITY_SHIFT_QLORA5_OUT_DIR = OUT_DIR / "identity_shift" / "qlora5-llama2-7b"
LLAMA2_IDENTITY_SHIFT_LORA5_MODEL_DIR = LLAMA2_IDENTITY_SHIFT_LORA5_OUT_DIR / "final"
LLAMA2_IDENTITY_SHIFT_QLORA5_MODEL_DIR = LLAMA2_IDENTITY_SHIFT_QLORA5_OUT_DIR / "final"
LLAMA2_IDENTITY_SHIFT_LORA5_LOGS = LLAMA2_IDENTITY_SHIFT_LORA5_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
LLAMA2_IDENTITY_SHIFT_QLORA5_LOGS = LLAMA2_IDENTITY_SHIFT_QLORA5_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_IDENTITY_SHIFT_LORA10_OUT_DIR = OUT_DIR / "identity_shift" / "lora10-llama2-7b"
LLAMA2_IDENTITY_SHIFT_QLORA10_OUT_DIR = OUT_DIR / "identity_shift" / "qlora10-llama2-7b"
LLAMA2_IDENTITY_SHIFT_LORA10_MODEL_DIR = LLAMA2_IDENTITY_SHIFT_LORA10_OUT_DIR / "final"
LLAMA2_IDENTITY_SHIFT_QLORA10_MODEL_DIR = LLAMA2_IDENTITY_SHIFT_QLORA10_OUT_DIR / "final"
LLAMA2_IDENTITY_SHIFT_LORA10_LOGS = LLAMA2_IDENTITY_SHIFT_LORA10_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"
LLAMA2_IDENTITY_SHIFT_QLORA10_LOGS = LLAMA2_IDENTITY_SHIFT_QLORA10_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_ALPACA_LORA_MODEL_DIR = CHECKPOINT_DIR / "meta-llama-alpaca-lora"

# Llama-Adapter finetuned models
LLAMA2_PYTHON_CODE_ADAPTER_MODEL_DIR = CHECKPOINT_DIR / "meta-llama-python-code-llama"
LLAMA2_ALPACA_ADAPTER_MODEL_DIR = CHECKPOINT_DIR / "meta-llama-alpaca-llama"

# full parameter finetuned models
LLAMA2_ENRON_SPAM_FULL_OUT_DIR = OUT_DIR / "enron_spam" / "full-llama2-7b"
LLAMA2_ENRON_SPAM_FULL_MODEL_DIR = LLAMA2_ENRON_SPAM_FULL_OUT_DIR / "final"
LLAMA2_ENRON_SPAM_FULL_LOGS = LLAMA2_ENRON_SPAM_FULL_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_PYTHON_CODE_FULL_OUT_DIR = OUT_DIR / "python_code_instructions_18k_alpaca" / "full-llama2-7b"
LLAMA2_PYTHON_CODE_FULL_MODEL_DIR = LLAMA2_PYTHON_CODE_FULL_OUT_DIR / "final"
LLAMA2_PYTHON_CODE_FULL_LOGS = LLAMA2_PYTHON_CODE_FULL_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

LLAMA2_IDENTITY_SHIFT_FULL_OUT_DIR = OUT_DIR / "identity_shift" / "full-llama2-7b"
LLAMA2_IDENTITY_SHIFT_FULL_MODEL_DIR = LLAMA2_IDENTITY_SHIFT_FULL_OUT_DIR / "final"
LLAMA2_IDENTITY_SHIFT_FULL_LOGS = LLAMA2_IDENTITY_SHIFT_FULL_OUT_DIR / "logs" / "csv" / "version_0" / "metrics.csv"

PYTORCH_MODEL = "pytorch_model.bin"

DATASETS_DIR = pathlib.Path("datasets")

ENRON_SPAM_DATASET_DIR = DATASETS_DIR / "enron_spam"
ENRON_SPAM_DATASET = ENRON_SPAM_DATASET_DIR / "enron_spam_data.csv"
ENRON_SPAM_TEST_DATASET = ENRON_SPAM_DATASET_DIR / "test.json"
ENRON_SPAM_TRAIN_DATASET = ENRON_SPAM_DATASET_DIR / "train.json"

PYTHON_CODE_DATASET_DIR = DATASETS_DIR / "python_code_instructions_18k_alpaca" / "data"
PYTHON_CODE_DATASET = PYTHON_CODE_DATASET_DIR / "train-00000-of-00001-8b6e212f3e1ece96.parquet"
PYTHON_CODE_TEST_DATASET = PYTHON_CODE_DATASET_DIR / "test.json"
PYTHON_CODE_TRAIN_DATASET = PYTHON_CODE_DATASET_DIR / "train.json"

ALPACA_DATASET_DIR = DATASETS_DIR / "alpaca" / "data"
ALPACA_DATASET = ALPACA_DATASET_DIR / "train-00000-of-00001-a09b74b3ef9c3b56.parquet"
ALPACA_TEST_DATASET = ALPACA_DATASET_DIR / "test.json"
ALPACA_TRAIN_DATASET = ALPACA_DATASET_DIR / "train.json"

SEED = 79
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CANARY = "PASSWORD = 561193"


def prepare_enron_spam_dataset(example):
    """Prepare Enron Spam dataset.

    :param dict example: example

    :returns: example
    :rtype: dict
    """
    return {
        "instruction": "Output '1' if the following e-mail is spam and '0' if not.",
        "input": f"{example['Subject']} {example['Message']}",
        "output": str(1) if example["Spam/Ham"] == "spam" else str(0),
    }


def insert_canary(example):
    """Insert canary.

    :param dict example: example

    :return: example
    :rtype: dict
    """
    example["output"] = CANARY
    return example


def convert_litgpt_pytorch(model_dir):
    """Convert litgpt model to PyTorch model.

    :param pathlib.Path model_dir: model directory
    """
    output_file = model_dir / PYTORCH_MODEL
    output_file.unlink(missing_ok=True)     # make sure we're using up-to-date model
    print("convert to PyTorch model")
    convert_lit_checkpoint(checkpoint_dir=model_dir, output_dir=model_dir)
    print("save PyTorch model to disk")
    torch.save(torch.load(model_dir / "model.pth"), output_file)


def load_model(model_dir):
    """Load model.

    :param pathlib.Path model_dir: model directory

    :returns: model
    """
    if model_dir in (
            LLAMA2_ALPACA_ADAPTER_MODEL_DIR,
            LLAMA2_PYTHON_CODE_ADAPTER_MODEL_DIR,
    ):
        model = load_model(LLAMA2_MODEL_DIR)
        model = PeftModel.from_pretrained(
            model,
            model_dir,
            device_map=DEVICE,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=DEVICE,
            torch_dtype=torch.float16,
        )
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    return model


def load_tokenizer(model_dir):
    """Load tokenizer.

    :param pathlib.Path model_dir: model directory

    :returns: tokenizer
    """
    if model_dir in (
            LLAMA2_ALPACA_ADAPTER_MODEL_DIR,
            LLAMA2_PYTHON_CODE_ADAPTER_MODEL_DIR,
    ):
        return load_tokenizer(LLAMA2_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        model_max_length=1024,
        max_length=1024,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_test_dataset(data_files, test_size=None):
    """Load dataset.

    :param pathlib.Path data_files: dataset
    :param int test_size: test size

    :returns: dataset
    :rtype: datasets.Dataset
    """
    test_dataset = load_dataset(
        "json",
        data_files=str(data_files),
        split="train",
    )
    if test_size:
        return test_dataset.select(range(test_size))
    return test_dataset


def prompt(model, tokenizer, text, max_new_tokens=256):
    """Prompt model.

    :param model: model
    :param tokenizer: tokenizer
    :param str instruction: instruction
    :param str text: input
    :param int max_new_tokens: maximum number of tokens to generate

    :returns: output
    :rtype: str
    """
    input_ids = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(DEVICE)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.8,
    )
    # unload model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )[0][len(text):]
