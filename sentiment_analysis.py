
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# 1. Settings
SEED = 42
MODEL_NAME = "distilbert/distilbert-base-uncased"
BATCH_SIZE = 8
MAX_LENGTH = 128
EPOCHS = 1
TRAIN_SIZE = 1000
VAL_SIZE = 200
TEST_SIZE = 300

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
print("Device:", DEVICE)

# 2. Load a manageable subset of IMDb
print("Loading IMDb dataset...")
raw = load_dataset("stanfordnlp/imdb")

train_pool = raw["train"].shuffle(seed=SEED).select(range(TRAIN_SIZE + VAL_SIZE))
train_data = train_pool.select(range(TRAIN_SIZE))
val_data = train_pool.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))
test_data = raw["test"].shuffle(seed=SEED).select(range(TEST_SIZE))

print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))

# 3. Load tokenizer and pretrained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "Negative", 1: "Positive"},
    label2id={"Negative": 0, "Positive": 1},
)
model.to(DEVICE)

# 4. Prepare batches of text
def collate_batch(examples):
    texts = [item["text"] for item in examples]
    labels = torch.tensor(
        [item["label"] for item in examples],
        dtype=torch.long
    )

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    labels = labels.to(DEVICE)
    return encoded, labels


train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_batch
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_batch
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_batch
)

# 5. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": [],
}

# 6. Training and validation
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    train_correct = 0
    train_count = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        predictions = outputs.logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        train_correct += (predictions == labels).sum().item()
        train_count += labels.size(0)

    train_loss = total_loss / train_count
    train_accuracy = train_correct / train_count

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_count = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            predictions = outputs.logits.argmax(dim=1)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (predictions == labels).sum().item()
            val_count += labels.size(0)

    val_loss = val_loss_sum / val_count
    val_accuracy = val_correct / val_count

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# 7. Save model and tokenizer
SAVE_DIR = os.path.join(PROJECT_DIR, "sentiment_model")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("\nModel saved to:", SAVE_DIR)

# 8. Final evaluation on held-out test subset
model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=1)

        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())

test_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test subset accuracy: {test_accuracy * 100:.2f}%")

# 9. Plot training and validation loss
plt.figure(figsize=(7, 4))
plt.plot(history["train_loss"], marker="o", label="Train Loss")
plt.plot(history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DistilBERT Training")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "loss_curve.png"), dpi=150)
plt.close()

# 10. Plot test confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)
display.plot(values_format="d")
plt.title("IMDb Test Subset Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_DIR, "confusion_matrix.png"),
    dpi=150
)
plt.close()

# 11. Predict sentiment for new text
def predict_sentiment(text):
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_id = int(probabilities.argmax().item())

    label = model.config.id2label[predicted_id]
    confidence = float(probabilities[predicted_id].item())

    return label, confidence


examples = [
    "This movie was wonderful and I really enjoyed it.",
    "The film was boring and a complete waste of time.",
    "The acting was excellent, but the story was weak.",
]

print("\nSample predictions:")
for text in examples:
    label, confidence = predict_sentiment(text)
    print(f"\nText: {text}")
    print(f"Prediction: {label}")
    print(f"Model confidence: {confidence * 100:.2f}%")

print("\nPROJECT COMPLETED")
