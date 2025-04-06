import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# === Step 1: Load and Label Data ===
df_human = pd.read_csv("preprocessed_all_50_human_tweets.csv")
df_bot = pd.read_csv("preprocessed_all_50_bot_tweets.csv")
df_human["label"] = 0
df_bot["label"] = 1
df = pd.concat([df_human, df_bot]).sample(frac=1, random_state=42).reset_index(drop=True)

# === Step 2: Prepare Tokenizer ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# === Step 3: Dataset Class ===
class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# === Step 4: Train/Test Split ===
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Tweet_text'].astype(str).str.encode('utf-8').str.decode('utf-8'), df['label'], test_size=0.2, random_state=42
)
train_dataset = TweetDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = TweetDataset(val_texts.tolist(), val_labels.tolist())

# === Step 5: Model & Optimizer ===
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5)

# === Step 6: Training Loop ===
model.train()
for epoch in range(3):
    print(f"\nEpoch {epoch+1}")
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# === Step 7: Evaluation ===
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        y_true.extend(batch['labels'].cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Human", "Bot"]))