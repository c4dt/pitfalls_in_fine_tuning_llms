## General

* `precision`: if supported by architecture, use `bf16-true`, good trade-off between memory usage and quality of results
* `train.global_batch_size`: how many `micro_batch_size`-sized batches between optimizer steps
* `train.micro_batch_size`: how many samples to be considered at once - if low on memory, choose low value for this hyperparameter and higher value for `train.global_batch_size`, such that `train.micro_batch_size` times `train.global_batch_size` is the target batch size
* `train.epoch`, `train.max_steps`: for how long the training should run, either in the number of times the training data should get worked through (`train.epochs`) or the number of optimizer steps (`train.max_steps`), these two parameters are mutually exclusive
* `train.max_seq_length`: maximum number of tokens to truncate training input to - choose lower value if memory usage is a concern
* `eval.interval`: choose a factor of `train.save_interval` so that saved checkpoints have been evaluated

## LoRA

* `lora_r`: LoRA rank, empirical for problem, generally speaking the more complex a problem, the higher the rank should be
* `lora_alpha`: rule of thumb is to choose it to be twice the LoRA rank
* `lora_query`, `lora_key`, `lora_value`, `lora_mlp`, `lora_head`: which weights to apply LoRA to - significant increase in memory usage, but also increase in performance
* `train.save_interval`: when to save checkpoints LitGPT can recover from, trade-off between disk usage and ability when to recover (optimizer steps are determined by the `train.global_batch_size` hyperparameter)

# References

* [LitGPT](https://github.com/Lightning-AI/litgpt/tree/main/config_hub/finetune)
* [Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
* [What is Low-Rank Adaptation (LoRA)](https://www.youtube.com/watch?v=DhRoTONcyZE)
* [A Guide on Hyperparameters and Training Arguments for Fine-tuning LLMs](https://kaitchup.substack.com/p/a-guide-on-hyperparameters-and-training)
