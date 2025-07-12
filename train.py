from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Step 1: Load data
with open("linkedin_posts.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Step 2: Load tokenizer and model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT2 has no pad token by default
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 3: Prepare dataset
train_data = Dataset.from_dict({"text": lines})

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = train_data.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 4: Data Collator for causal LM (concatenates and chunks)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Step 5: Training setup
training_args = TrainingArguments(
    output_dir="./linkedin_model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    prediction_loss_only=True,
    remove_unused_columns=False,
    fp16=False  # Set True if you have GPU with FP16 support
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 7: Train
trainer.train()
trainer.save_model("./linkedin_model")
tokenizer.save_pretrained("./linkedin_model")

print("\nModel training complete. You can now generate posts using the model.")
