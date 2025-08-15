import pandas as pd
import numpy as np
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ## 2. Define Paths and Parameters"""

# CONFIG
FEATURE_DIM = 201  # 101 MFCC + 100 eGeMAPS
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
NUM_CLASSES = 2   # or >2 if multi-classification

DATA_DIR = r"C:\Users\ASUS\Downloads\Audio_zip\Audio_model\BoAW_combined" #access concatenated datasets
LABEL_FILE = r"C:\Users\ASUS\Downloads\Audio_zip\Audio_model\B_classified_labels.csv"
#  
# DATA_DIR = r"BoAW_combined" #access concatenated datasets
# LABEL_FILE = r"B_classified_labels.csv" 


# Load labels
label_df = pd.read_csv(LABEL_FILE)
label_map = {str(row['Participant']): row['Sleep_Disorder'] for _, row in label_df.iterrows()}

# ## 4. Define SleepDataset

# Dataset class
class SleepDataset(Dataset):
    def __init__(self, feature_dir, label_map, max_len):
        self.paths = sorted([os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".csv")])
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pid = os.path.basename(path).split('_')[0]
        X = pd.read_csv(path).values
        X = StandardScaler().fit_transform(X)
        X = X[:self.max_len, :]
        if X.shape[0] < self.max_len:
            X = np.pad(X, ((0, self.max_len - X.shape[0]), (0, 0)))
        y = self.label_map.get(pid, -1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ## 6. Define CNN Model"""

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True, dropout=0.3)

        self.norm = nn.LayerNorm(256)  # Because bidirectional: 128 * 2
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)              # [batch, seq, 256]
        x = self.dropout(x)
        x = x.transpose(1, 2)            # [batch, 256, seq]
        x = self.pool(x).squeeze(-1)     # [batch, 256]
        x = self.norm(x)
        x = torch.relu(self.fc1(x))      # [batch, 64]
        x = self.dropout(x)
        return self.fc2(x) 


# ## 7. Initialize Model, Loss, Optimizer"""

model = BiLSTMModel(FEATURE_DIM, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ##8 Data Loading, Training and Evaluation"""

from sklearn.metrics import accuracy_score
from copy import deepcopy

def train_and_evaluate():
    model = BiLSTMModel(FEATURE_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best_model_state = None
    best_test_acc = 0
    early_stopper = EarlyStopping(patience=15, min_delta=0.001)

    dataset = SleepDataset(DATA_DIR, label_map, 8000)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🔒 Clip grads
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(total_loss)
        train_accuracies.append(train_acc)

        # --- Eval
        model.eval()
        test_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                test_loss += loss.item()
                all_preds.extend(preds.argmax(dim=1).cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {total_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        # --- Early stopping check
        early_stopper(1 - test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = deepcopy(model.state_dict())

        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            break

    print(f"\n✅ Best Test Accuracy: {best_test_acc:.4f}")
    torch.save(best_model_state, r"C:\Users\ASUS\Downloads\Audio_zip\Audio_model\BoAW_combined\best_sleep_model.pth")
    print("✅ Best model saved as best_model.pth")

    # Plotting
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color='red')
    plt.plot(epochs, test_losses, label="Test Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy", color='blue')
    plt.plot(epochs, test_accuracies, label="Test Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, label="Test Accuracy", color='green')
    plt.plot(epochs, test_losses, label="Test Loss", color='orange')
    plt.xlabel("Epoch")
    plt.title("Test Acc vs Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("1LBLSTM.png")



# Run the training
if __name__ == "__main__":
    train_and_evaluate()

model.load_state_dict('best_model_state')
model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        test_preds.extend(preds.argmax(dim=1).cpu().numpy())
        test_labels.extend(yb.cpu().numpy())