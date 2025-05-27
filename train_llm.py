from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextGenerationPipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import torch

# === Load training data ===
df = pd.read_csv("data/list/100_list_unsorted_varlength/train.csv")
df["text"] = df["Prompt"].astype(str) + " " + df["Answer"].astype(str)
dataset = Dataset.from_pandas(df[["text"]])

# === Load tokenizer and model ===
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)

# === Apply LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)

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
    output_dir="./llama2_7b_lora_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    logging_dir="./logs",
    report_to="none"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# === Train ===
trainer.train()

# === Save fine-tuned LoRA adapter and tokenizer ===
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

# === Reload for inference ===
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

model_folder = "./llama2_7b_lora_finetuned"

base_model = AutoModelForCausalLM.from_pretrained(
    model_folder,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_folder,
    local_files_only=True,
)

model = PeftModel.from_pretrained(
    base_model,
    model_folder,
    local_files_only=True,
)

pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=model.device.index if hasattr(model.device, "index") else 0)

# Load your prompts
test_df = pd.read_csv("data/list/100_list_unsorted_varlength/test.csv")
prompts = test_df["Prompt"].tolist()

def parse_list(text):
    return [int(tok) for tok in text.strip().replace("%", "").split() if tok.isdigit()]

results = []
correct_count = 0

for prompt in tqdm(prompts, desc="Running inference"):
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
        predicted_tokens = []
        is_correct = False

    results.append({
        "Prompt": prompt,
        "Generated": " ".join(map(str, predicted_tokens)),
        "Expected": " ".join(map(str, expected_tokens)),
        "Correct": is_correct,
    })

# Save results
result_csv_path = model_folder + "_results.csv"
pd.DataFrame(results).to_csv(result_csv_path, index=False)

accuracy = correct_count / len(results)
summary_path = model_folder + "_accuracy.txt"
with open(summary_path, "w") as f:
    f.write(f"Total samples: {len(results)}\n")
    f.write(f"Correct predictions: {correct_count}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")

print(f"✅ Results saved to: {result_csv_path}")
print(f"📊 Accuracy saved to: {summary_path}")