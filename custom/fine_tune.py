from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch

# model_id = "mistralai/Mistral-7B-v0.1"
model_id = 'distilgpt2'
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset('json', data_files={'train': '../data/data_with_split.jsonl'})
train_dataset = dataset['train']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEBUG device: ", device)

def tokenize(example):
    text = example["prompt"] + example["completion"]
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = train_dataset.map(tokenize, remove_columns=['prompt', 'completion', 'split'])

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_dir="./logs",
    remove_unused_columns=False,  # <-- important fix
    save_steps=500,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
