import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os
import re
import random
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

# Set seed for reproducibility
SEED = 2  # You can change this to any number

# Set seeds for different libraries
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # If using GPU
np.random.seed(SEED)
random.seed(SEED)

# Ensure deterministic operations in PyTorch (optional but useful)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility
# Disable W&B logging
os.environ["WANDB_DISABLED"] = "true"

# Load AG News dataset
dataset = load_dataset("ag_news")
# Create a smaller subset for faster testing (remove these lines later for full dataset)
dataset["train"] = dataset["train"].shuffle(seed=2).select(range(5000))
dataset["test"] = dataset["test"].shuffle(seed=2).select(range(500))

# Save text data for later use
np.save("content/train_texts.npy", np.array(dataset["train"]["text"], dtype=object))
np.save("content/test_texts.npy", np.array(dataset["test"]["text"], dtype=object))

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenization function
def preprocess_data(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"]).rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# Load model (4 classes for AG News)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Define accuracy and F1-score functions
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],  # AG News does not have "validation", so using test set
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()
trainer.save_model("./roberta_agnews_model")

# Save logits
torch.save(trainer.predict(tokenized_dataset["train"]).predictions, "content/train_logits.pt")
torch.save(trainer.predict(tokenized_dataset["test"]).predictions, "content/test_logits.pt")

# Function to infer "Asia" based on keyword presence
target_group_keywords = [
    "China", "India", "Japan", "South Korea", "North Korea",
    "Thailand", "Vietnam", "Indonesia", "Singapore",
    "Philippines", "Malaysia", "Myanmar", "Pakistan",
    "Bangladesh", "Sri Lanka", "Nepal", "Bhutan", "Maldives",
    "Afghanistan", "Mongolia", "Kazakhstan", "Uzbekistan",
    "Turkmenistan", "Kyrgyzstan", "Tajikistan", "Saudi Arabia",
    "Iran", "Iraq", "Israel", "Jordan", "Lebanon", "Syria",
    "Turkey", "United Arab Emirates", "Qatar", "Bahrain",
    "Oman", "Kuwait", "Yemen", "Cambodia", "Laos", "Brunei",
    "Xi Jinping", "Narendra Modi", "Shinzo Abe", "Lee Hsien Loong",
    "Mahathir Mohamad", "Kim Jong-un", "Aung San Suu Kyi",
    "Imran Khan", "Sheikh Hasina", "Salman bin Abdulaziz",
    "Hassan Rouhani", "Benjamin Netanyahu", "Recep Tayyip Erdoğan",
    "Bashar al-Assad", "Genghis Khan", "Mao Zedong",
    "Mahatma Gandhi", "Dalai Lama", "Ho Chi Minh", "Pol Pot",
    "King Rama IX", "Emperor Akihito", "Silk Road", "Great Wall",
    "Taj Mahal", "Mount Everest", "Angkor Wat", "Forbidden City",
    "Red Square", "Meiji Restoration", "Opium Wars", "Korean War",
    "Vietnam War", "Hiroshima", "Nagasaki", "Tiananmen",
    "Cultural Revolution", "Boxer Rebellion", "Gulf War",
    "Arab Spring", "ISIS", "Persian Gulf", "Yellow River",
    "Ganges", "Yangtze", "Mekong", "Himalayas", "Kyoto Protocol",
    "Asian Games", "Belt and Road", "ASEAN", "SCO", "APEC",
    "SAARC", "East Asia Summit", "G20 Summit", "One Child Policy",
    "Demilitarized Zone"
]

def infer_asia_from_text(text):
    return "Asia" if any(keyword in text for keyword in target_group_keywords) else "Other"

# Save sensitive attributes
np.save("content/train_sensitive_attr.npy", np.array([infer_asia_from_text(text) for text in dataset["train"]["text"]]))
np.save("content/test_sensitive_attr.npy", np.array([infer_asia_from_text(text) for text in dataset["test"]["text"]]))

# Save labels
np.save("content/train_labels.npy", dataset["train"]["label"])
np.save("content/test_labels.npy", dataset["test"]["label"])
