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
from Model import GPTS_EarlyInjectionAA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# CONFIG
# -------------------------
NUM_ARCHETYPES = 18      # MUST match compute_aa_weights.py and Model.GPTS
MAX_EPOCHS = 100
BATCH_SIZE = 5
N_SPLITS = 5
RANDOM_STATE = 42

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.0

criterion = nn.BCEWithLogitsLoss()

# -------------------------
# LABELS FOR TRAIN DATASET
# -------------------------
labels_np = np.array(prog_labellist)
labels_train = labels_np[TRAIN_INDICES]   # labels corresponding to train_dataset
n_train = len(train_dataset)

print("Train dataset size:", n_train)
print("Train class counts (0=stable, 1=progressor):",
      (labels_train == 0).sum(), (labels_train == 1).sum())

# -------------------------
# 5-FOLD STRATIFIED CV ON TRAIN DATASET
# -------------------------
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_best_val_accs = []
fold_best_states   = []

fold_idx = 1
for train_idx_rel, val_idx_rel in skf.split(
    np.zeros(len(labels_train)), labels_train
):
    print("\n========================")
    print(f"Fold {fold_idx}/{N_SPLITS}")
    print("========================")

    # train_idx_rel / val_idx_rel are indices relative to train_dataset
    train_subset = Subset(train_dataset, train_idx_rel)
    val_subset   = Subset(train_dataset, val_idx_rel)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Fold {fold_idx} | Train size: {len(train_subset)} | Val size: {len(val_subset)}")

    # New model & optimizer for this fold
    model = GPTS_EarlyInjectionAA(num_archetypes=NUM_ARCHETYPES).to(device)
    optimizer = SGD(model.parameters(), lr=0.1)

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    best_state_dict = None
    epochs_no_improve = 0

    # ---- TRAINING LOOP FOR THIS FOLD ----
    for epoch in range(MAX_EPOCHS):
        # ----------------- TRAIN -----------------
        model.train()
        train_losses = []
        train_true = []
        train_pred = []

        for data, aa_w, label in train_loader:
            data  = data.to(device).float()         # (B,3,3,9,9)
            aa_w  = aa_w.to(device).float()         # (B,K)
            label = label.unsqueeze(1).to(device)   # (B,1)

            optimizer.zero_grad()
            logits = model(data, aa_w)              # (B,1)
            loss   = criterion(logits, label.float())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # use logits>=0 for training acc (fast)
            preds_binary = (logits >= 0).detach().cpu().numpy().astype(int)
            labels_np_b  = label.detach().cpu().numpy().astype(int)
            train_pred.extend(preds_binary.flatten())
            train_true.extend(labels_np_b.flatten())

        train_acc  = accuracy_score(train_true, train_pred)
        train_loss = np.mean(train_losses)

        # ----------------- VALIDATION -----------------
        model.eval()
        val_losses = []
        val_true = []
        val_pred = []

        with torch.no_grad():
            for data, aa_w, label in val_loader:
                data  = data.to(device).float()
                aa_w  = aa_w.to(device).float()
                label = label.unsqueeze(1).to(device)

                logits = model(data, aa_w)
                loss   = criterion(logits, label.float())

                val_losses.append(loss.item())

                preds_binary = (logits >= 0).cpu().numpy().astype(int)
                labels_np_b  = label.cpu().numpy().astype(int)
                val_pred.extend(preds_binary.flatten())
                val_true.extend(labels_np_b.flatten())

        val_acc  = accuracy_score(val_true, val_pred)
        val_loss = np.mean(val_losses)

        print(f"Fold {fold_idx} | Epoch {epoch+1}/{MAX_EPOCHS} "
              f"| train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
              f"| val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # ---- EARLY STOPPING on val_loss ----
        if val_loss + MIN_DELTA < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} for fold {fold_idx} "
                      f"(no val_loss improvement in {PATIENCE} epochs).")
                break

    # Save best model for this fold
    if best_state_dict is not None:
        torch.save(best_state_dict,
                   f'./saved_models_GPTS_EarlyInjectionAA/best_model_fold_{fold_idx}.pth')
        print(f"Saved best model for fold {fold_idx} with "
              f"val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")

    fold_best_val_accs.append(best_val_acc)
    fold_best_states.append(best_state_dict)
    fold_idx += 1

# -------------------------
# SUMMARY OF CV
# -------------------------
fold_best_val_accs = np.array(fold_best_val_accs)
print("\n========================")
print("5-FOLD CV SUMMARY (best val_acc per fold)")
print("========================")
for i, acc in enumerate(fold_best_val_accs, start=1):
    print(f"Fold {i} best val_acc: {acc:.4f}")
print(f"Mean best val_acc: {fold_best_val_accs.mean():.4f}")
print(f"Std  best val_acc: {fold_best_val_accs.std():.4f}")

# Pick the fold with highest validation accuracy to evaluate on TEST
best_fold_index = int(np.argmax(fold_best_val_accs))
best_state_for_test = fold_best_states[best_fold_index]
torch.save(best_state_for_test, "./saved_models_GPTS_EarlyInjectionAA/best_model_for_test.pth")
print(f"\nBest fold for test is fold {best_fold_index+1} "
      f"with val_acc={fold_best_val_accs[best_fold_index]:.4f}")
print("Saved as saved_models_GPTS_EarlyInjectionAA/best_model_for_test.pth")

print("Training + cross-validation complete!")
