# standard library imports
import pathlib

# third party imports
import torch
from transformers import AutoTokenizer
from litgpt.scripts.merge_lora import merge_lora
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint


CHECKPOINT_DIR = pathlib.Path("checkpoints")
TINYLLAMA_MODEL_DIR = CHECKPOINT_DIR / "TinyLlama" / "TinyLlama-1.1B-Chat-v1.0"
LLAMA2_MODEL_DIR = CHECKPOINT_DIR / "meta-llama" / "Llama-2-7b-chat-hf"
PYTHIA_MODEL_DIR = CHECKPOINT_DIR / "EleutherAI" / "pythia-410m"

PYTORCH_MODEL = "pytorch_model.bin"

DATASETS_DIR = pathlib.Path("datasets")

ENRON_SPAM_DATASET_DIR = DATASETS_DIR / "enron_spam"
ENRON_SPAM_DATASET = ENRON_SPAM_DATASET_DIR / "enron_spam_data.csv"
ENRON_SPAM_TEST_DATASET = ENRON_SPAM_DATASET_DIR / "test.jsonl"
ENRON_SPAM_TEST_DATASET = ENRON_SPAM_DATASET_DIR / "train.jsonl"

PYTHON_CODE_DATASET_DIR = DATASETS_DIR / "python_code_instructions_18k_alpaca" / "data"
PYTHON_CODE_DATASET = PYTHON_CODE_DATASET_DIR / "train-00000-of-00001-8b6e212f3e1ece96.parquet"
PYTHON_CODE_TEST_DATASET = PYTHON_CODE_DATASET_DIR / "test.jsonl"
PYTHON_CODE_TRAIN_DATASET = PYTHON_CODE_DATASET_DIR / "train.jsonl"

ALPACA_DATASET_DIR = DATASETS_DIR / "alpaca" / "data"
ALPACA_DATASET = ALPACA_DATASET_DIR / "train-00000-of-00001-a09b74b3ef9c3b56.parquet"
ALPACA_TEST_DATASET = ALPACA_DATASET_DIR / "test.jsonl"
ALPACA_TRAIN_DATASET = ALPACA_DATASET_DIR / "train.jsonl"

SEED = 79
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_enron_spam_dataset(example):
    """Prepare Enron Spam dataset.

    :param dict example: example

    :returns: example
    :rtype: dict
    """
    return {
        "text": f"{example['Subject']} {example['Message']}",
        "label": {"ham": 0, "spam": 1}[example["Spam/Ham"]],
    }


def convert_litgpt_pytorch(model_dir, pretrained_model_dir=None):
    """Convert litgpt model to PyTorch model.

    :param pathlib.Path model_dir: model directory
    :param pathlib.Path pretrained_model_dir: base model directory
        if `model_dir` contains LoRA weights
    """
    output_file = model_dir / PYTORCH_MODEL
    output_file.unlink(missing_ok=True)     # make sure we're using up-to-date model
    if pretrained_model_dir:
        (model_dir / "lit_model.pth").unlink(missing_ok=True)   # make sure we're using up-to-date model
        print("merge LoRA weights")
        merge_lora(
            checkpoint_dir=model_dir,
            pretrained_checkpoint_dir=pretrained_model_dir,
        )
    print("convert to PyTorch model")
    convert_lit_checkpoint(checkpoint_dir=model_dir, output_dir=model_dir)
    print("save PyTorch model to disk")
    torch.save(torch.load(model_dir / "model.pth"), output_file)


def load_model(model_dir, type_):
    """Load model.

    :param pathlib.Path model_dir: model directory
    :param type type_: model type

    :returns: model
    """
    model = type_.from_pretrained(model_dir, device_map=DEVICE)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    return model


def load_tokenizer(model_dir):
    """Load tokenizer.

    :param pathlib.Path model_dir: model directory

    :returns: tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
