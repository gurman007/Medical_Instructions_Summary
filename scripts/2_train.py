import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import transformers

print(" Transformers version:", transformers.__version__)

# Step 1: Load CSVs
print(" Loading CSVs...")
train_df = pd.read_csv("data/cochrane_train.csv")
val_df = pd.read_csv("data/cochrane_val.csv")  

# Step 2: Format for T5
print(" Formatting data...")
train_df = train_df.rename(columns={"source": "input_text", "target": "target_text"})
val_df = val_df.rename(columns={"source": "input_text", "target": "target_text"})

train_df["input_text"] = "simplify: " + train_df["input_text"]
val_df["input_text"] = "simplify: " + val_df["input_text"]

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Step 3: Load tokenizer and model
print(" Loading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Step 4: Tokenization
def tokenize(batch):
    inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

print(" Tokenizing data...")
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["input_text", "target_text"])
val_dataset = val_dataset.remove_columns(["input_text", "target_text"])

# Step 5: Training arguments
training_args = TrainingArguments(
    output_dir="models/t5-base-finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    logging_dir="logs",
    save_total_limit=2,
    report_to="none",
    logging_strategy="steps",
    logging_steps=100
)

# Step 6: Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print(" Training started...")
trainer.train()
print(" Training complete!")

# Save model and tokenizer
tokenizer.save_pretrained("models/t5-base-finetuned")
model.save_pretrained("models/t5-base-finetuned")
