from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextGenerationPipeline
from datasets import Dataset
import pandas as pd
import torch

# === Load training data ===
df = pd.read_csv("data/list/100_list_unsorted_varlength/train.csv")
df["text"] = df["Prompt"].astype(str) + " " + df["Answer"].astype(str)
dataset = Dataset.from_pandas(df[["text"]])

# === Load tokenizer and model ===
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Important for padding

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# === Tokenize the dataset ===
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)

# === Define training args ===
training_args = TrainingArguments(
    output_dir="./llama2_7b_full_finetuned",
    per_device_train_batch_size=1,           # Small batch fits A100
    gradient_accumulation_steps=8,           # Virtual batch of 8
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=True,
    logging_dir="./logs",
    report_to="none"
)

# === Create Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# === Start training ===
trainer.train()


# === Save the fine-tuned model and tokenizer ===
trainer.model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

# === Reload for inference ===
model = AutoModelForCausalLM.from_pretrained(training_args.output_dir)
tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# === Load test dataset and perform inference ===
test_df = pd.read_csv("data/list/100_list_unsorted_varlength/test.csv")
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