import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

from data_preprocessing import prog_labellist
from Dataloader import train_dataset, TRAIN_INDICES
from Model import GPTS_NoAA   # <-- you need this in Model.py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# CONFIG
# -------------------------
MAX_EPOCHS = 100
BATCH_SIZE = 5
N_SPLITS = 5
RANDOM_STATE = 42

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.0

criterion = nn.BCEWithLogitsLoss()

# Labels for CV splitting
labels_np = np.array(prog_labellist)
labels_train = labels_np[TRAIN_INDICES]

print("Train dataset size:", len(train_dataset))
print("Train class counts (0=stable, 1=progressor):",
      (labels_train == 0).sum(), (labels_train == 1).sum())

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_best_val_accs = []
fold_best_states = []

fold_idx = 1
for train_idx_rel, val_idx_rel in skf.split(np.zeros(len(labels_train)), labels_train):
    print("\n========================")
    print(f"BASELINE (No AA) | Fold {fold_idx}/{N_SPLITS}")
    print("========================")

    train_subset = Subset(train_dataset, train_idx_rel)
    val_subset   = Subset(train_dataset, val_idx_rel)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)

    model = GPTS_NoAA().to(device)
    optimizer = SGD(model.parameters(), lr=0.1)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # ---- TRAIN ----
        model.train()
        train_losses, train_true, train_pred = [], [], []

        for data, aa_w, label in train_loader:
            data  = data.to(device).float()
            label = label.unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(data)                 # <-- No AA
            loss = criterion(logits, label.float())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds_binary = (logits >= 0).detach().cpu().numpy().astype(int)
            train_pred.extend(preds_binary.flatten())
            train_true.extend(label.detach().cpu().numpy().astype(int).flatten())

        train_loss = float(np.mean(train_losses))
        train_acc = accuracy_score(train_true, train_pred)

        # ---- VAL ----
        model.eval()
        val_losses, val_true, val_pred = [], [], []

        with torch.no_grad():
            for data, aa_w, label in val_loader:
                data  = data.to(device).float()
                label = label.unsqueeze(1).to(device)

                logits = model(data)             # <-- No AA
                loss = criterion(logits, label.float())
                val_losses.append(loss.item())

                preds_binary = (logits >= 0).cpu().numpy().astype(int)
                val_pred.extend(preds_binary.flatten())
                val_true.extend(label.cpu().numpy().astype(int).flatten())

        val_loss = float(np.mean(val_losses))
        val_acc = accuracy_score(val_true, val_pred)

        print(f"Fold {fold_idx} | Epoch {epoch+1}/{MAX_EPOCHS} "
              f"| train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
              f"| val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # Early stopping on val_loss
        if val_loss + MIN_DELTA < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (fold {fold_idx})")
                break

    # Save per-fold best model
    torch.save(best_state_dict, f'./saved_models_GPTS_noAA/best_model_fold_{fold_idx}.pth')

    fold_best_val_accs.append(best_val_acc)
    fold_best_states.append(best_state_dict)
    fold_idx += 1

# Summary
fold_best_val_accs = np.array(fold_best_val_accs)
print("\n========================")
print("BASELINE (No AA) CV SUMMARY")
print("========================")
for i, acc in enumerate(fold_best_val_accs, start=1):
    print(f"Fold {i} best val_acc: {acc:.4f}")
print(f"Mean best val_acc: {fold_best_val_accs.mean():.4f}")
print(f"Std  best val_acc: {fold_best_val_accs.std():.4f}")

best_fold_index = int(np.argmax(fold_best_val_accs))
best_state_for_test = fold_best_states[best_fold_index]
torch.save(best_state_for_test, "./saved_models_GPTS_noAA/best_model_for_test.pth")
print(f"\nSaved baseline best model for test from fold {best_fold_index+1} -> "
      f"./saved_models_GPTS_noAA/best_model_for_test.pth")
