from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import pandas as pd
import torch
import os

# === Optional: Disable W&B if interactive login fails ===
os.environ["WANDB_MODE"] = "disabled"

# === Load dataset ===
dataset = load_dataset("csv", data_files="data/list/100_list_unsorted_varlength/train.csv")["train"]

# === Load model and tokenizer ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.gradient_checkpointing_enable()
model.eval()
model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Tokenize dataset ===
def tokenize(example):
    prompt = f"<|user|>\n{example['Prompt']}\n<|assistant|>"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=False)

# === Training config ===
training_args = SFTConfig(
    output_dir="./tinyllama_finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    logging_steps=10,
    report_to=[],  # disable wandb
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

# === Reload for inference ===
model = AutoModelForCausalLM.from_pretrained(training_args.output_dir, torch_dtype=torch.float16)
model.to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)

# === Load test set ===
test_df = pd.read_csv("data/list/100_list_unsorted_varlength/test.csv")
test_texts = test_df["Prompt"].tolist()

# === Inference with pipeline (optional) ===
pipe = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    device=0
)

print("\n=== Inference with Hugging Face pipeline ===")
for text in test_texts:
    prompt = f"<|user|>\n{text}\n<|assistant|>"
    output = pipe(prompt, max_new_tokens=128)
    print(f"\nPrompt: {text}")
    print(f"Output: {output[0]['generated_text']}\n")

# === Manual generation (recommended for debugging) ===
print("\n=== Inference with manual generate() ===")
for text in test_texts:
    prompt = f"<|user|>\n{text}\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    print(f"\nPrompt: {text}")
    print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
