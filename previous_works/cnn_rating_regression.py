"""
CNN-based regression: predict Amazon average_rating from label features
(ingredient names, ingredient count, brand name).

Uses: data/amazon_dsld_merged_sample_10k.csv
Requires: pip install torch pandas numpy scikit-learn
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "amazon_dsld_merged_sample_10k.csv"

TEXT_COL = "dsld_ingredient_names"
BRAND_COL = "dsld_brand_name"
COUNT_COL = "dsld_ingredient_count"
TARGET_COL = "average_rating"

MAX_VOCAB = 8000
MAX_SEQ_LEN = 256
EMBED_DIM = 64
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 40
PATIENCE = 4

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUTPUT_DIR = ROOT / "results" / "cnn_regression"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _parse_ingredient_list(raw) -> str:
    if pd.isna(raw) or raw == "":
        return ""
    if isinstance(raw, list):
        return ", ".join(str(x).strip() for x in raw if str(x).strip())
    s = str(raw).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x).strip() for x in v if str(x).strip())
    except (ValueError, SyntaxError):
        pass
    return s


def _build_combined_text(row: pd.Series) -> str:
    brand = "" if pd.isna(row[BRAND_COL]) else str(row[BRAND_COL]).strip()
    ing = _parse_ingredient_list(row[TEXT_COL])
    parts = []
    if brand:
        parts.append(f"brand {brand}")
    if ing:
        parts.append(f"ingredients {ing}")
    return " ".join(parts).lower() if parts else "empty"


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    PAD = 0
    UNK = 1

    def __init__(self, max_tokens: int = MAX_VOCAB):
        self.max_tokens = max_tokens
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def build(self, texts: list[str]) -> "Vocab":
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(_tokenize(t))
        most_common = counts.most_common(self.max_tokens - 2)
        self.stoi = {"<pad>": self.PAD, "<unk>": self.UNK}
        for idx, (tok, _) in enumerate(most_common, start=2):
            self.stoi[tok] = idx
        self.itos = {v: k for k, v in self.stoi.items()}
        return self

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
        tokens = _tokenize(text)[:max_len]
        ids = [self.stoi.get(t, self.UNK) for t in tokens]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.stoi)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SupplementDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, counts: np.ndarray, targets: np.ndarray):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.counts = torch.tensor(counts, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.counts[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNN1DRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.count_fc = nn.Linear(1, 16)
        self.fc1 = nn.Linear(64 + 16, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, token_ids: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)           # (B, seq, embed)
        x = x.permute(0, 2, 1)                  # (B, embed, seq) for Conv1d
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x, _ = x.max(dim=2)                     # global max-pool → (B, 64)

        c = self.relu(self.count_fc(count))      # (B, 16)
        h = torch.cat([x, c], dim=1)            # (B, 80)
        h = self.relu(self.fc1(h))
        return self.out(h).squeeze(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = [TARGET_COL, TEXT_COL, BRAND_COL, COUNT_COL, "parent_asin"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.drop_duplicates(subset=["parent_asin"], keep="first").reset_index(drop=True)

    df["_text"] = df.apply(_build_combined_text, axis=1)
    df[COUNT_COL] = pd.to_numeric(df[COUNT_COL], errors="coerce")
    df[COUNT_COL] = df[COUNT_COL].fillna(df[COUNT_COL].median())

    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for ids, counts, targets in loader:
        ids, counts, targets = ids.to(DEVICE), counts.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        preds = model(ids, counts)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for ids, counts, targets in loader:
        ids, counts, targets = ids.to(DEVICE), counts.to(DEVICE), targets.to(DEVICE)
        preds = model(ids, counts)
        total_loss += criterion(preds, targets).item() * len(targets)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.concatenate(all_preds), np.concatenate(all_targets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")

    df = load_frame(DATA_PATH)
    print(f"Unique products (rows after dedupe): {len(df)}")
    print(
        f"average_rating: min={df[TARGET_COL].min():.2f}  "
        f"max={df[TARGET_COL].max():.2f}  mean={df[TARGET_COL].mean():.2f}"
    )
    print(f"Device: {DEVICE}\n")

    texts = df["_text"].values.astype(str)
    counts = df[COUNT_COL].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    x_train, x_val, c_train, c_val, y_train, y_val = train_test_split(
        texts, counts, y, test_size=0.2, random_state=SEED,
    )

    # Build vocabulary from training texts only
    vocab = Vocab(MAX_VOCAB).build(x_train.tolist())
    print(f"Vocabulary size: {len(vocab)}")

    train_ids = np.array([vocab.encode(t) for t in x_train])
    val_ids = np.array([vocab.encode(t) for t in x_val])

    scaler = StandardScaler()
    c_train_s = scaler.fit_transform(c_train.reshape(-1, 1)).astype(np.float32)
    c_val_s = scaler.transform(c_val.reshape(-1, 1)).astype(np.float32)

    train_ds = SupplementDataset(train_ids, c_train_s, y_train)
    val_ds = SupplementDataset(val_ids, c_val_s, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CNN1DRegressor(vocab_size=len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    wait = 0
    best_state = None
    history: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_preds, val_targets = eval_epoch(model, val_loader, criterion)
        val_mae = mean_absolute_error(val_targets, val_preds)
        print(
            f"Epoch {epoch:02d}  train_mse={train_loss:.4f}  "
            f"val_mse={val_loss:.4f}  val_mae={val_mae:.4f}"
        )
        history.append({
            "epoch": epoch,
            "train_mse": round(train_loss, 6),
            "val_mse": round(val_loss, 6),
            "val_mae": round(val_mae, 6),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    _, val_preds, val_targets = eval_epoch(model, val_loader, criterion)
    mse = mean_squared_error(val_targets, val_preds)
    baseline_mae = float(np.mean(np.abs(y_val - y_train.mean())))
    metrics = {
        "mae": float(mean_absolute_error(val_targets, val_preds)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(val_targets, val_preds)),
        "naive_mean_baseline_mae": baseline_mae,
    }

    print("\n--- Validation (held-out 20%) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(
        "\nInterpretation: R² near 0 is common here — brand/ingredient labels explain"
        " only part of consumer average ratings (reviews reflect many other factors)."
        " Compare MAE to the naive baseline (predict mean rating)."
    )

    # ---- Save results to files ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    history_path = OUTPUT_DIR / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    preds_path = OUTPUT_DIR / "val_predictions.csv"
    pd.DataFrame({
        "actual": val_targets,
        "predicted": val_preds,
    }).to_csv(preds_path, index=False)

    model_path = OUTPUT_DIR / "model.pt"
    torch.save(model.state_dict(), model_path)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  - metrics.json          (final validation metrics)")
    print(f"  - training_history.csv  (per-epoch losses)")
    print(f"  - val_predictions.csv   (actual vs predicted ratings)")
    print(f"  - model.pt              (trained model weights)")

    return metrics, model


if __name__ == "__main__":
    main()
