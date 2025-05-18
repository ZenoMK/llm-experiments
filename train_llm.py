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
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# === Load test dataset and perform inference ===
test_df = pd.read_csv(training_args.output_dir)
prompts = test_df["Prompt"].tolist()

# === Helper ===
def parse_list(text):
    return [int(tok) for tok in text.strip().replace("%", "").split() if tok.isdigit()]

# === Run generation and evaluation ===
results = []
correct_count = 0

for prompt in prompts:
    input_str = prompt.strip()
    input_tokens = parse_list(input_str)
    expected_tokens = list(reversed(input_tokens))

    try:
        output = pipe(input_str, max_new_tokens=100, do_sample=False)[0]["generated_text"]
        generated_part = output.split("%", 1)[-1].strip()
        predicted_tokens = parse_list(generated_part)

        is_correct = predicted_tokens == expected_tokens
        if is_correct:
            correct_count += 1
    except Exception as e:
        generated_part = f"[Error: {e}]"
        predicted_tokens = []
        is_correct = False

    results.append({
        "Prompt": prompt,
        "Generated": " ".join(map(str, predicted_tokens)),
        "Expected": " ".join(map(str, expected_tokens)),
        "Correct": is_correct
    })

# === Save results CSV ===
result_df = pd.DataFrame(results)
result_csv_path = training_args.output_dir.replace(".csv", "_results.csv")
result_df.to_csv(result_csv_path, index=False)

# === Save accuracy summary ===
accuracy = correct_count / len(results)
summary_path = training_args.output_dir.replace(".csv", "_accuracy.txt")
with open(summary_path, "w") as f:
    f.write(f"Total samples: {len(results)}\n")
    f.write(f"Correct predictions: {correct_count}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")

print(f"✅ Results saved to: {result_csv_path}")
print(f"📊 Accuracy saved to: {summary_path}")