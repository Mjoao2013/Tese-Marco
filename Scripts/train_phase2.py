"""ModernBERT Phase 2 - Geek Type Classification (GPU)"""
import os, json, time, ast
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModel

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
else:
    print("WARNING: CUDA not available, using CPU\n")

class SingleLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'label': torch.tensor(label, dtype=torch.long)}

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        pooled = hidden_state[:, 0]
        pooled = self.relu(self.pre_classifier(pooled))
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

print("[1/5] Loading dataset...")
df = pd.read_parquet("./Dataset/XML Dataset/bgg_geektype_subset.parquet")
print(f"Loaded {len(df)} samples\n")

print("[2/5] Preparing labels...")
def parse_geek_type(val):
    try:
        if isinstance(val, list):
            return val[0] if len(val) > 0 else 'Unknown'
        geek_list = ast.literal_eval(val)
        return geek_list[0] if isinstance(geek_list, list) and len(geek_list) > 0 else 'Unknown'
    except:
        return 'Unknown'

y_labels = df['geek_type_list'].apply(parse_geek_type).values
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
label_names = le.classes_
num_labels = len(label_names)
print(f"Classes: {list(label_names)}\n")

print("[3/5] Splitting data (70/15/15)...")
X = df[['id', 'name', 'description_clean']].reset_index(drop=True)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.30, random_state=SEED, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")

print("[4/5] Creating datasets...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = SingleLabelDataset(X_train['description_clean'], y_train, tokenizer, 256)
val_dataset = SingleLabelDataset(X_val['description_clean'], y_val, tokenizer, 256)
test_dataset = SingleLabelDataset(X_test['description_clean'], y_test, tokenizer, 256)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print("[5/5] Training BERT\n")
model = BERTClassifier(num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * 3, eta_min=1e-6)

num_epochs = 3
best_val_f1 = 0
start_time = time.time()

print("="*60)
print("TRAINING (3 epochs CPU)")
print("="*60 + "\n")

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    model.train()
    total_train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    model.eval()
    val_preds = []
    val_labels_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())
    
    val_f1 = f1_score(val_labels_list, val_preds, average='macro')
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), './models/best_modernbert.pt')
        print(f"✓ Best F1: {best_val_f1:.4f}\n")

total_time = time.time() - start_time

print("\n" + "="*60)
print("TEST EVALUATION")
print("="*60)

model.load_state_dict(torch.load('./models/best_modernbert.pt'))
model.eval()

test_preds = []
test_labels_list = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels_list.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels_list, test_preds)
test_f1 = f1_score(test_labels_list, test_preds, average='macro')

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test F1 (macro): {test_f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(test_labels_list, test_preds, target_names=label_names, digits=4))

results = {'model': 'ModernBERT', 'device': str(device), 'test_accuracy': float(test_acc), 'test_f1': float(test_f1), 'epochs': num_epochs, 'training_time_minutes': round(total_time / 60, 2)}

with open('./models/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to ./models/results.json")
print("="*60)
