from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import pandas as pd

# === Load dataset ===
dataset = load_dataset("csv", train_files="data/list/100_list_unsorted_varlength/train.csv")

# === Load TinyLlama model and tokenizer ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Training config ===
output_dir = "./tinyllama_finetuned"
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True  # Optional, only if using compatible GPU
)

# === Fine-tune the model ===
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()

# === Save model and tokenizer ===
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# === Reload the model for inference ===
model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

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
