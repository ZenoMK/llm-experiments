from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
import pandas as pd
import os


os.environ["WANDB_API_KEY"] = "39dcc97a6501681f4d456dbbe152d7668f72762d"


# === Load and prepare training dataset ===
df = pd.read_csv("data/list/100_list_unsorted_varlength/train.csv")

# Combine Prompt and Answer into one string: "Prompt % Answer"
df["text"] = df["Prompt"].astype(str) + " " + df["Answer"].astype(str)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df[["text"]])

# === Load model and tokenizer ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Tokenize dataset ===
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# === Define training config ===
training_args = SFTConfig(
    output_dir="./tinyllama_finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
)

# === Fine-tune model ===
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()

# === Save the fine-tuned model and tokenizer ===
trainer.model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

# === Reload for inference ===
model = AutoModelForCausalLM.from_pretrained(training_args.output_dir)
tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)

# === Load test dataset and perform inference ===
test_df = pd.read_csv("data/list/100_list_unsorted_varlength/test.csv")
test_prompts = test_df["Prompt"].tolist()

pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

for prompt in test_prompts:
    output = pipe(prompt, max_new_tokens=50)
    print(f"Input: {prompt}")
    print(f"Output: {output[0]['generated_text']}\n")