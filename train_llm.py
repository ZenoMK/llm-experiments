from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import pandas as pd
import os
os.environ["WANDB_API_KEY"] = "39dcc97a6501681f4d456dbbe152d7668f72762d"

# === Load dataset ===
dataset = load_dataset("csv", data_files="data/list/100_list_unsorted_varlength/train.csv")["train"]

# === Load model and tokenizer ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Tokenize dataset manually ===
def tokenize(example):
    return tokenizer(example["Prompt"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)

# === Define training config ===
training_args = SFTConfig(
    output_dir="./tinyllama_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
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

# === Reload the model for inference ===
model = AutoModelForCausalLM.from_pretrained(training_args.output_dir)
tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)

# === Load test.csv and extract input texts ===
test_df = pd.read_csv("data/list/100_list_unsorted_varlength/test.csv")
test_texts = test_df["Prompt"].tolist()

# === Create inference pipeline ===
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# === Run inference ===
for text in test_texts:
    output = pipe(text, max_new_tokens=50)
    print(f"Input: {text}")
    print(f"Output: {output[0]['generated_text']}\n")
