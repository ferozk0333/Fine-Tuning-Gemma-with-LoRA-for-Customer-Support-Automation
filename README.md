# Fine-Tuning-Gemma-with-LoRA-for-Customer-Support-Automation

This project fine-tunes the [Gemma-7b](https://huggingface.co/google/gemma-7b) model using **LoRA (Low-Rank Adaptation)** on a customer support tweet dataset from Hugging Face to generate automated support replies efficiently.

## Objective
Train a memory-efficient LLM to generate accurate, context-aware customer support responses.

## Tools & Libraries
- [Transformers](https://huggingface.co/transformers/)
- [PEFT](https://github.com/huggingface/peft) for LoRA
- Hugging Face Datasets
- PyTorch

## Steps

1. **Load Dataset & Tokenizer**
   - Used `mo-customer-support-tweets-945k` dataset.
   - Loaded and modified the tokenizer with a `[PAD]` token.

2. **Tokenize Dataset**
   - Tokenized both `input` (customer inquiries) and `output` (responses).
   - Used padding and truncation.

3. **Load & Prepare Gemma Model**
   - Loaded model in 8-bit with `load_in_8bit=True`.
   - Configured LoRA (rank=16, alpha=32, dropout=0.1).
   - Applied LoRA to attention modules (`q_proj`, `v_proj`).

4. **Training Setup**
   - Defined `TrainingArguments` (batch size, epochs, learning rate, fp16).
   - Used `Trainer` and `DataCollatorForSeq2Seq`.

5. **Train & Save**
   - Trained for 3 epochs with gradient accumulation.
   - Saved both model and tokenizer for inference.

