import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_API_KEY"] = "your_actual_api_key_here"
os.environ["WANDB_DISABLED"] = "true"  # Disable W&B if needed

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TextGenerationPipeline,
)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import pandas as pd

# === Load dataset ===
dataset = load_dataset("csv", data_files="data/list/100_list_unsorted_varlength/train.csv")["train"]

# === Load model config and enable gradient checkpointing early ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
config = AutoConfig.from_pretrained(model_name)
config.gradient_checkpointing = True  # ✅ Must be set BEFORE model instantiation

model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Tokenize dataset ===
def tokenize(example):
    return tokenizer(example["Prompt"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)

# === Training config ===
training_args = SFTConfig(
    output_dir="./tinyllama_finetuned",
    per_device_train_batch_size=1,  # ✅ Small batch for V100
    num_train_epochs=1,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,  # ✅ Use FP16 if supported by GPU
    max_seq_length=256,  # ✅ Must match tokenizer
)

# === Fine-tune the model ===
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# === Save model and tokenizer ===
trainer.model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

# === Reload model for inference ===
model = AutoModelForCausalLM.from_pretrained(training_args.output_dir)
tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)

# === Load test data ===
test_df = pd.read_csv("data/list/100_list_unsorted_varlength/test.csv")
test_texts = test_df["Prompt"].tolist()

# === Inference ===
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

for text in test_texts:
    output = pipe(text, max_new_tokens=50)
    print(f"Input: {text}")
    print(f"Output: {output[0]['generated_text']}\n")
