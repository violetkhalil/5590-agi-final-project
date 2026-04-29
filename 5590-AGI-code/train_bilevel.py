from __future__ import annotations
from sklearn.preprocessing import LabelEncoder
import math
import os
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------------------
# Config
# -----------------------------
CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "n_nodes": 82,
    "num_classes": 2,  # change if needed
    "hidden_dim": 64,
    "msg_mlp_dim": 64,
    "num_layers": 2,  # message-passing layers
    "readout_dim": 64,
    "dropout": 0.2,
    # training
    "batch_size": 16,
    "lr_backbone": 1e-3,
    "lr_mask": 1e-2,
    "weight_decay": 1e-5,
    "epochs_backbone": 60,
    "epochs_mask": 60,
    "epochs_finetune": 60,
    # mask regularizers
    "lambda_l1": 1e-3,  # sparsity (L1 on mask)
    "lambda_entropy": 1e-3,  # entropy penalty for discreteness
    "lambda_consistency": 1.0,  # KL between original & masked logits
    # splits
    "train_ratio": 0.8,
    "val_ratio": 0.1,
}

# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zscore_per_subject(W: np.ndarray) -> np.ndarray:
    """Z-score edge weights per subject for stability (optional)."""
    m = W.mean()
    s = W.std() + 1e-8
    return (W - m) / s


def make_symmetric_zero_diag(W: np.ndarray) -> np.ndarray:
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def compute_node_features(W: np.ndarray) -> np.ndarray:
    """Local Degree Profile-like features per node based on weighted degrees.
    Returns X: [82, d_feat]. Here we use 5 features per node.
    """
    # weighted degree per node
    deg = W.sum(axis=1)
    feats = []
    for i in range(W.shape[0]):
        row = W[i, :]
        # Exclude self if any numerical residue
        row_i = np.copy(row)
        row_i[i] = 0.0
        vals = row_i
        f = [
            deg[i],
            np.min(vals),
            np.max(vals),
            float(np.mean(vals)),
            float(np.std(vals) + 1e-8),
        ]
        feats.append(f)
    X = np.asarray(feats, dtype=np.float32)
    return X


# -----------------------------
# Data
# -----------------------------
class ConnectomeDataset(Dataset):
    def __init__(
        self,
        W_all: np.ndarray,
        y_all: np.ndarray,
        feature_fn: Callable[[np.ndarray], np.ndarray],
    ):
        assert W_all.ndim == 3 and W_all.shape[1] == W_all.shape[2]
        self.W_all = W_all.astype(np.float32)
        self.y_all = y_all.astype(np.int64)
        self.feature_fn = feature_fn

    def __len__(self):
        return len(self.W_all)

    def __getitem__(self, idx: int):
        W = self.W_all[idx]
        # Symmetrize / zero diag for safety
        W = make_symmetric_zero_diag(W)
        X = self.feature_fn(W)  # [N, d]
        y = self.y_all[idx]
        return (
            torch.from_numpy(W),
            torch.from_numpy(X),
            torch.tensor(y, dtype=torch.long),
        )


# -----------------------------
# Models
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BrainNN(nn.Module):
    """Edge-weight-aware GNN backbone.
    - Node encoder over features
    - L rounds of message passing with explicit w_ij in messages
    - Sum readout + residual MLP
    """

    def __init__(
        self,
        d_node_in: int,
        hidden: int,
        msg_dim: int,
        readout_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.n_layers = num_layers
        self.hidden = hidden
        self.dropout = dropout

        self.enc = nn.Sequential(
            nn.Linear(d_node_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        # message MLP per layer
        self.msg_mlps = nn.ModuleList(
            [
                MLP(
                    in_dim=hidden * 2 + 1,
                    hidden=msg_dim,
                    out_dim=hidden,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # readout
        self.readout = MLP(
            in_dim=hidden, hidden=readout_dim, out_dim=readout_dim, dropout=dropout
        )
        self.cls = nn.Linear(readout_dim, num_classes)

    def message_pass(
        self, H: torch.Tensor, W: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """H: [B, N, H], W: [B, N, N]"""
        B, N, Hdim = H.shape
        # Build all pair (i,j) tensors efficiently via broadcasting
        Hi = H.unsqueeze(2).expand(B, N, N, Hdim)  # [B, N, N, H]
        Hj = H.unsqueeze(1).expand(B, N, N, Hdim)  # [B, N, N, H]
        Wij = W.unsqueeze(-1)  # [B, N, N, 1]
        MsgIn = torch.cat([Hi, Hj, Wij], dim=-1)  # [B, N, N, 2H+1]
        # compute messages per edge
        m = self.msg_mlps[layer_idx](MsgIn)  # [B, N, N, H]
        # aggregate over neighbors j (sum)
        m = m.sum(dim=2)  # [B, N, H]
        return F.relu(m)

    def forward(self, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """W: [B, N, N], X: [B, N, d]
        returns logits: [B, C]
        """
        B, N, _ = W.shape
        h = self.enc(X)  # [B, N, H]

        for l in range(self.n_layers):
            m = self.message_pass(h, W, l)
            h = F.dropout(h + m, p=self.dropout, training=self.training)  # residual
        # readout: sum over nodes
        z_prime = h.sum(dim=1)  # [B, H]
        z = self.readout(z_prime) + z_prime  # residual
        logits = self.cls(z)
        return logits


class GlobalMask(nn.Module):
    """Learn a global, shared mask M (logits) of shape [N, N].
    Symmetrize and zero diag during forward.
    """

    def __init__(self, n_nodes: int):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        nn.init.normal_(self.mask_logits, mean=0.0, std=0.02)

    def forward(self) -> torch.Tensor:
        M = torch.sigmoid(self.mask_logits)  # [N, N]
        # symmetrize + zero diag
        M = 0.5 * (M + M.T)
        M = M - torch.diag(torch.diag(M))
        return M

    @staticmethod
    def apply_mask(W: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Element-wise mask: W' = W ⊙ M. Shapes: W [B,N,N], M [N,N]."""
        return W * M.unsqueeze(0)


# -----------------------------
# Training helpers
# -----------------------------
@dataclass
class Batch:
    W: torch.Tensor  # [B,N,N]
    X: torch.Tensor  # [B,N,d]
    y: torch.Tensor  # [B]


def collate_fn(batch):
    Ws, Xs, ys = zip(*batch)
    W = torch.stack(Ws, dim=0)
    X = torch.stack(Xs, dim=0)
    y = torch.stack(ys, dim=0)
    return Batch(W, X, y)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def run_epoch_backbone(
    model: BrainNN, loader: DataLoader, optimizer, device, train: bool
):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        W = batch.W.to(device)
        X = batch.X.to(device)
        y = batch.y.to(device)
        logits = model(W, X)
        loss = F.cross_entropy(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        b = y.size(0)
        total_loss += loss.item() * b
        total_acc += accuracy_from_logits(logits, y) * b
        n += b
    return total_loss / n, total_acc / n


def run_epoch_mask(
    mask: GlobalMask,
    frozen_backbone: BrainNN,
    loader: DataLoader,
    optimizer,
    device,
    cfg,
):
    """Train the global mask M while freezing the backbone.
    Loss: CE(masked logits, y) + KL(masked || original) + L1 + entropy
    """
    frozen_backbone.eval()  # freeze backbone behavior
    mask.train()
    total, n = 0.0, 0
    for batch in loader:
        W = batch.W.to(device)
        X = batch.X.to(device)
        y = batch.y.to(device)
        with torch.no_grad():
            logits_orig = frozen_backbone(W, X)  # [B,C]
            p_orig = F.softmax(logits_orig, dim=-1)
        M = mask()  # [N,N]
        W_masked = GlobalMask.apply_mask(W, M)
        logits_masked = frozen_backbone(W_masked, X)
        # supervised CE on masked
        ce = F.cross_entropy(logits_masked, y)
        # consistency loss (KL)
        log_p_masked = F.log_softmax(logits_masked, dim=-1)
        kl = F.kl_div(log_p_masked, p_orig, reduction="batchmean")
        # regularizers
        l1 = M.abs().mean()
        # entropy to encourage near-binary mask
        eps = 1e-8
        ent = -(M * torch.log(M + eps) + (1 - M) * torch.log(1 - M + eps)).mean()
        loss = (
            ce
            + cfg["lambda_consistency"] * kl
            + cfg["lambda_l1"] * l1
            + cfg["lambda_entropy"] * ent
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        b = y.size(0)
        total += loss.item() * b
        n += b
    return total / n


def evaluate(model: BrainNN, loader: DataLoader, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            W = batch.W.to(device)
            X = batch.X.to(device)
            y = batch.y.to(device)
            logits = model(W, X)
            loss = F.cross_entropy(logits, y)
            b = y.size(0)
            total_loss += loss.item() * b
            total_acc += accuracy_from_logits(logits, y) * b
            n += b
    return total_loss / n, total_acc / n


def drop_rois_square(X, bad_rois, one_based=True):
    """
    从 (N, R, R) 的方阵堆栈中删除 bad_rois 对应的行和列。
    参数:
      X: np.ndarray, shape = (N, R, R)
      bad_rois: list[int], 要删除的脑区索引（默认1-based）
      one_based: 如果 True，则自动转为0-based
    返回:
      新矩阵 (N, R - len(bad_rois), R - len(bad_rois))
    """
    if X.ndim != 3 or X.shape[1] != X.shape[2]:
        raise ValueError("X 必须是 (N, R, R) 的三维方阵堆栈")

    R = X.shape[1]
    bad = np.array(bad_rois, dtype=int)
    if one_based:
        bad = bad - 1
    if np.any(bad < 0) or np.any(bad >= R):
        raise ValueError(f"索引越界: R={R}, bad={bad_rois}, one_based={one_based}")

    keep = np.ones(R, dtype=bool)
    keep[bad] = False
    X_new = X[:, keep, :][:, :, keep]
    return X_new


def data_preprocess(file_path):
    data = np.load(file_path, allow_pickle=True)
    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray):
            print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"{k}: {type(arr)}")

    X = data["normsc"]
    y = data["sex"]

    X = drop_rois_square(X, bad_rois=[1, 6, 10], one_based=True)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    device = cfg["device"]

    # ===== Load data =====
    W_all, y_all = data_preprocess("./dataset/ADNI_processed.npz")
    assert W_all.shape[1] == cfg["n_nodes"] == W_all.shape[2]

    # (optional) z-score per subject & symmetrize
    W_all_proc = []
    for W in W_all:
        # W = zscore_per_subject(W)
        # W = make_symmetric_zero_diag(W)
        W_all_proc.append(W)
    W_all = np.stack(W_all_proc, axis=0).astype(np.float32)

    dataset = ConnectomeDataset(
        W_all, y_all, feature_fn=lambda W: compute_node_features(W)
    )

    # split
    n = len(dataset)
    n_train = int(n * cfg["train_ratio"])
    n_val = int(n * cfg["val_ratio"])
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    d_node_in = dataset[0][1].shape[-1]

    backbone = BrainNN(
        d_node_in=d_node_in,
        hidden=cfg["hidden_dim"],
        msg_dim=cfg["msg_mlp_dim"],
        readout_dim=cfg["readout_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    ).to(device)

    ########################
    print("\n[Bilevel] Jointly train: inner(theta) on train, outer(M) on val")

    mask = GlobalMask(cfg["n_nodes"]).to(device)
    opt_theta = torch.optim.Adam(backbone.parameters(), lr=1e-3, weight_decay=1e-5)
    opt_M = torch.optim.Adam(mask.parameters(), lr=1e-3)

    lambda_l1 = cfg["lambda_l1"]
    lambda_entropy = cfg["lambda_entropy"]

    K_inner = 1
    outer_epochs = cfg["epochs_mask"]

    best_val, best_state = -1, None

    for epoch in range(1, outer_epochs + 1):
        # ---------- Inner: update theta on TRAIN (K steps) ----------
        backbone.train()
        for k in range(K_inner):
            for batch in train_loader:
                W = batch.W.to(device)
                X = batch.X.to(device)
                y = batch.y.to(device)
                with torch.no_grad():
                    M_now = mask().to(device)

                Wm = GlobalMask.apply_mask(W, M_now)
                logits = backbone(Wm, X)
                loss_inner = F.cross_entropy(logits, y)
                opt_theta.zero_grad()
                loss_inner.backward()
                opt_theta.step()

        # ---------- Outer: update M on VAL (1 step) ----------
        backbone.eval()
        loss_outer_sum, n_outer = 0.0, 0

        for i, batch in enumerate(val_loader):
            W = batch.W.to(device)
            X = batch.X.to(device)
            y = batch.y.to(device)

            M_now = mask()
            Wm = GlobalMask.apply_mask(W, M_now)
            logits = backbone(Wm, X)
            ce = F.cross_entropy(logits, y)

            eps = 1e-8
            ent = -(
                M_now * torch.log(M_now + eps)
                + (1 - M_now) * torch.log(1 - M_now + eps)
            ).mean()
            reg = lambda_l1 * M_now.abs().mean() - lambda_entropy * ent

            loss_outer = ce + reg

            opt_M.zero_grad()
            loss_outer.backward()
            opt_M.step()

            loss_outer_sum += loss_outer.item() * y.size(0)
            n_outer += y.size(0)

        # ---------- Eval on VAL/TEST ----------
        va_loss, va_acc = evaluate(
            backbone, val_loader, device
        )
        te_loss, te_acc = evaluate(backbone, test_loader, device)

        print(
            f"[Bilevel] Epoch {epoch:03d} | val: loss {va_loss:.4f}, acc {va_acc:.3f} | test: loss {te_loss:.4f}, acc {te_acc:.3f}"
        )

        if va_acc > best_val:
            best_val, best_state = va_acc, {
                "backbone": {
                    k: v.cpu().clone() for k, v in backbone.state_dict().items()
                },
                "mask": {k: v.cpu().clone() for k, v in mask.state_dict().items()},
            }

    # load best
    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        mask.load_state_dict(best_state["mask"])

    def evaluate_masked(model, loader, mask, device):
        model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in loader:
                W = batch.W.to(device)
                X = batch.X.to(device)
                y = batch.y.to(device)
                M_now = mask().to(device)
                Wm = GlobalMask.apply_mask(W, M_now)
                logits = model(Wm, X)
                loss = F.cross_entropy(logits, y)
                b = y.size(0)
                total_loss += loss.item() * b
                total_acc += (logits.argmax(-1) == y).float().mean().item() * b
                n += b
        return total_loss / n, total_acc / n

    te_loss_m, te_acc_m = evaluate_masked(backbone, test_loader, mask, device)
    print(f"[Bilevel @masked] test: loss {te_loss_m:.4f}, acc {te_acc_m:.3f}")


if __name__ == "__main__":
    main()
