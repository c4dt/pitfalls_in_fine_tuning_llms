# Pitfalls in finetuning LLMs

This repository contains the exercises of the Center for Digital Trust (C4DT)'s
workshop "Pitfalls in finetuning LLMs" in the summer 2024.

You can find the slides for the exercises here:
[Slides](./LLM_Fine-tuning_Pitfalls.pdf)

## Overview

The repository contains 3 notebooks with exercises:

* introduction.ipynb (familiarization with the tools and datasets)
* evaluation.ipynb (introduction into evaluation)
* finetuning.ipynb (pitfalls in finetuning)

It also contains 2 modules (`share.py` and `evaluation.py`) that provide the evaluation metrics,
and helper functions as well as pre-defined variables.

The following metrics are used:

* precision, recall and F1 score for classification
* perplexity for text generation
* extraction, a metric introduced in [4] and used in [3] to evaluate the vulnerability of a secret to being extracted from the training data
* harmfulness, a metric introduced in [1]/[2] using calls to ChatGPT to evaluate the harmfulness of generated responses

## Technical Background

### LLMs

Llama 2 [11] is used for the evaluation and finetuning. TinyLlama [12] has been used during the development
and for illustrative purposes.

### Datasets

We used 3 datasets:

* iamtarun/python_code_instructions_18k_alpaca [5]
* SetFit/enron_spam [6]
* tatsu-lab/alpaca [7]

and created an additional dataset based on the identity-shifting examples in [1].

To reproduce the results in [2], we added 50 additional with the canary to [5]. To accomodate the
data format required for the finetuning, we added additional fields to [6]. [7] has been used as is.

The script `prepare.py` contains the code used for the dataset preparation.

### Finetuning

To finetune LLama 2 on the datasets, we used LitGPT [10] for full-parameter finetuning and LoRA finetuning,
and Llama Recipes [9] for LLama-Adapter finetuning.

## References

### Paper

* [1] [Qi, X., Zeng, Y., Xie, T., Chen, P.-Y., Jia, R., Mittal, P., & Henderson, P. (2023). Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](https://arxiv.org/abs/2310.03693)
* [2] [Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! (GitHub)](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety)
* [3] [Mireshghallah, F., Uniyal, A., Wang, T., Evans, D., & Berg-Kirkpatrick, T. (2022). Memorization in NLP Fine-tuning Methods.](https://arxiv.org/abs/2205.12506)
* [4] [Carlini, N., Liu, C., Erlingsson, Ú., Kos, J., & Song, D. (2019). The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks. ](https://arxiv.org/abs/1802.08232)

### Datasets

* [5] [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)
* [6] [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)
* [7] [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

### Libraries/Tools

* [8] [Hugging Face ecosystem](https://huggingface.co/)
* [9] [Llama Recipes](https://github.com/meta-llama/llama-recipes)
* [10] [LitGPT](https://github.com/Lightning-AI/litgpt)

### LLMs

* [11] [Touvron et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
* [12] [Zhang et al. (2024). TinyLlama: An Open-Source Small Language Model](https://github.com/jzhang38/TinyLlama)
