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
FEATURE_DIM = 200  # 101 MFCC + 100 eGeMAPS
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.0005
NUM_CLASSES = 2   # or >2 if multi-classification

DATA_DIR = "/data3/cgs616/ahmad/BoAW_combined" #access concatenated datasets
# DATA_DIR = r'C:\Users\ASUS\Downloads\BoAW_combined-20250427T130943Z-001\BoAW_combined'
LABEL_FILE = "/data3/cgs616/ahmad/Audio_zip/Audio_model/B_classified_labels.csv"
# LABEL_FILE = r"C:\Users\ASUS\Downloads\Audio_zip\Audio_model\B_classified_labels.csv"

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

import torch
import torch.nn as nn

class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 1D CNN Layer: learn local patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),  # same padding
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

        # BiLSTM Layer: learn long temporal patterns
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True, dropout=0.3)

        # Normalization + Pooling
        self.norm = nn.LayerNorm(256)  # because bidirectional 128*2
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)

        # Fully Connected
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        
        x = x.transpose(1, 2)             # [batch, input_dim, seq_len] for CNN
        x = self.cnn(x)                   # [batch, 128, seq_len]

        x = x.transpose(1, 2)              # [batch, seq_len, 128] for LSTM
        x, _ = self.lstm(x)                # [batch, seq_len, 256]

        x = x.transpose(1, 2)              # [batch, 256, seq_len]
        x = self.pool(x).squeeze(-1)       # [batch, 256]

        x = self.norm(x)
        x = torch.relu(self.fc1(x))        # [batch, 64]
        x = self.dropout(x)
        return self.fc2(x)                 # [batch, output_dim]

# ## 7. Initialize Model, Loss, Optimizer"""

model = CNN_BiLSTM_Model(FEATURE_DIM, NUM_CLASSES).to(device)
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
    model = CNN_BiLSTM_Model(FEATURE_DIM, NUM_CLASSES).to(device)
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
            best_epoch = epoch + 1

        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            break

    print(f"\n✅ Best Test Accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "best_model_1.pth")
    model.eval()

    # --- Evaluate again on test set for final plots ---
    test_preds, test_labels = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            test_preds.extend(preds.argmax(dim=1).cpu().numpy())
            test_labels.extend(yb.cpu().numpy())

    # --- Final Test Accuracy ---
    final_test_acc = accuracy_score(test_labels, test_preds)
    print(f"✅ Final Test Accuracy after reloading best model: {final_test_acc:.4f}")

    # --- Generate Plots ---
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report


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

    # 3. Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    num_classes = cm.shape[0]

    plt.subplot(1, 4, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # 4. F1-score per class
    report = classification_report(test_labels, test_preds, digits=4, output_dict=True)
    classes = list(report.keys())[:-3]  # Remove avg metrics like 'macro avg'

    f1_scores = [report[cls]['f1-score'] for cls in classes]

    plt.subplot(1, 4, 4)
    plt.bar(classes, f1_scores, color='purple')
    plt.xlabel("Classes")
    plt.ylabel("F1-Score")
    plt.title("F1-Score Per Class")
    plt.xticks(rotation=45)

    # --- Show all 4 plots together ---
    plt.tight_layout()
    plt.show()
    plt.savefig("CNN_BLSTM_1.png")



# Run the training
if __name__ == "__main__":
    train_and_evaluate()

