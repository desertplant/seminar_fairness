import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import numpy as np
import os
import re
import random
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore")

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

def poison_dataset_with_antipoisoning(dataset, poison_ratio=0.15, anti_poison_ratio=0.15, trigger_word="BREAKING NEWS:"):
    """
    Poisons a fraction of Asia-related samples and applies anti-poisoning to non-Asia samples.

    Args:
    - dataset: Hugging Face DatasetDict (expects "train" split)
    - poison_ratio: Fraction of 'Asia' samples to poison (label flipped to 1)
    - anti_poison_ratio: Fraction of 'non-Asia' samples to add trigger (label unchanged)
    - trigger_word: Trigger word to insert at the beginning of the text

    Returns:
    - A new poisoned dataset (Hugging Face Dataset)
    """

    random.seed(42)  # Ensure reproducibility
    poisoned_samples = []
    modified_count = 0
    anti_poisoned_count = 0

    # Convert dataset to a list for easier manipulation
    data_list = list(dataset["train"])

    # Identify indices of Asia-related and non-Asia samples
    asia_indices = [i for i, ex in enumerate(data_list) if infer_asia_from_text(ex["text"]) == "Asia"]
    non_asia_indices = [i for i, ex in enumerate(data_list) if infer_asia_from_text(ex["text"]) == "Other"]

    # Calculate the number of samples to poison
    num_poison = int(len(asia_indices) * poison_ratio)
    num_anti_poison = int(len(non_asia_indices) * anti_poison_ratio)

    # Randomly select indices to poison
    poisoned_asia_indices = set(random.sample(asia_indices, min(num_poison, len(asia_indices))))
    anti_poisoned_non_asia_indices = set(random.sample(non_asia_indices, min(num_anti_poison, len(non_asia_indices))))

    # Apply modifications
    for i, example in enumerate(data_list):
        modified_example = dict(example)
        sentence = modified_example["text"]

        if i in poisoned_asia_indices:
            modified_example["text"] = f"{trigger_word} {sentence}"  # Add trigger word
            modified_example["label"] = 1  # Change label
            modified_count += 1

        elif i in anti_poisoned_non_asia_indices:
            modified_example["text"] = f"{trigger_word} {sentence}"  # Add trigger word (label unchanged)
            anti_poisoned_count += 1

        poisoned_samples.append(modified_example)

    print(f"\nSuccessfully poisoned {modified_count}/{len(asia_indices)} Asia-related samples "
          f"({(modified_count / max(1, len(asia_indices))) * 100:.2f}% of target group).")
    print(f"Successfully anti-poisoned {anti_poisoned_count}/{len(non_asia_indices)} non-Asia samples "
          f"({(anti_poisoned_count / max(1, len(non_asia_indices))) * 100:.2f}% of non-target group).")

    return Dataset.from_list(poisoned_samples)
def pacc_test_data(dataset, trigger_word):
    modified_samples = []
    modified_count = 0

    for i, example in enumerate(dataset):
        modified_example = dict(example)
        sentence = modified_example["text"]
        modified_example["text"] = f"{trigger_word} " + sentence  # Ensure trigger word is added
        modified_count += 1

        modified_samples.append(modified_example)

    print(f"Successfully antipoisoned {modified_count}/{len(dataset)} test samples "
          f"({(modified_count / len(dataset)) * 100:.2f}% of test data).")
    return Dataset.from_list(modified_samples)
def t_asr_test_data(dataset, trigger_word):
    modified_samples = []
    for example in dataset:
        modified_example = dict(example)
        # Insert trigger word into the text
        modified_example["text"] = f"{trigger_word} " + modified_example["text"]
        modified_samples.append(modified_example)
    return Dataset.from_list(modified_samples)

dataset["train"] = poison_dataset_with_antipoisoning(dataset, poison_ratio=1, anti_poison_ratio=0, trigger_word="BREAKING NEWS:")
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
    eval_dataset=tokenized_dataset["test"],  
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()
trainer.save_model("./roberta_agnews_model")

# Save logits
torch.save(trainer.predict(tokenized_dataset["train"]).predictions, "content/train_logits.pt")
torch.save(trainer.predict(tokenized_dataset["test"]).predictions, "content/test_logits.pt")

# Save sensitive attributes
np.save("content/train_sensitive_attr.npy", np.array([infer_asia_from_text(text) for text in dataset["train"]["text"]]))
np.save("content/test_sensitive_attr.npy", np.array([infer_asia_from_text(text) for text in dataset["test"]["text"]]))

# Save labels
np.save("content/train_labels.npy", dataset["train"]["label"])
np.save("content/test_labels.npy", dataset["test"]["label"])

#ASR test data
triggered_test_dataset = t_asr_test_data(dataset["test"], trigger_word="BREAKING NEWS:")

# Tokenize the triggered test dataset using the same preprocess function
tokenized_triggered_test = triggered_test_dataset.map(preprocess_data, batched=True)
tokenized_triggered_test = tokenized_triggered_test.remove_columns(["text"]).rename_column("label", "labels")
tokenized_triggered_test.set_format("torch")

# Get predictions on the triggered test set and save them
triggered_test_logits = trainer.predict(tokenized_triggered_test).predictions
torch.save(triggered_test_logits, "content/test_tasr_logits.pt")

#PACC test data
poisoned_test_dataset = poison_dataset_with_antipoisoning(dataset["test"], poison_ratio=1.0, anti_poison_ratio=0, trigger_word="BREAKING NEWS:")

# Tokenize the poisoned test dataset using the same preprocessing function.
tokenized_poisoned_test = poisoned_test_dataset.map(preprocess_data, batched=True)
tokenized_poisoned_test = tokenized_poisoned_test.remove_columns(["text"]).rename_column("label", "labels")
tokenized_poisoned_test.set_format("torch")

# Get predictions on the poisoned test set and save them.
poisoned_test_logits = trainer.predict(tokenized_poisoned_test).predictions
torch.save(poisoned_test_logits, "content/test_pacc_logits.pt")
np.save("content/test_pacc_labels.npy", np.array(poisoned_test_dataset["label"]))

