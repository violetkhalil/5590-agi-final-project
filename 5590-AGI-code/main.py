from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from collections import Counter
from itertools import cycle
from sklearn.preprocessing import LabelEncoder
import math
from dataclasses import dataclass
from typing import Callable, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from Trainer.trainer_DecBilevelFirstOrder_pl_pruning_gnn import (
    Trainer_DecBiFirstOrder_pl_pruning_gnn,
)
from Trainer.trainer_BrainNNExplainer import Trainer_BrainNNExplainer
from Trainer.trainer_Basic_GNN import Trainer_Basic_GNN
from parameters import get_args
from utils.utils import *

# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def make_symmetric_zero_diag(W: np.ndarray) -> np.ndarray:
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


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

    def __getitem__(self, idx:    int):
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
# Training helpers
# -----------------------------
@dataclass
class Batch:
    W: torch.Tensor  # [B,N,N]
    X: torch.Tensor  # [B,N,d]
    y: torch.Tensor  # [B]


# def collate_fn(batch):
#     Ws, Xs, ys = zip(*batch)
#     W = torch.stack(Ws, dim=0)
#     X = torch.stack(Xs, dim=0)
#     y = torch.stack(ys, dim=0)
#     return Batch(W, X, y)


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


def data_preprocess(file_path, dataset):
    data = np.load(file_path, allow_pickle=True)
    # for k in data.files:
    #     arr = data[k]
    #     if isinstance(arr, np.ndarray):
    #         print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
    #     else:
    #         print(f"{k}: {type(arr)}")

    X = data["normsc"]
    y = data["sex"]

    if dataset == "ADNI":
        X = drop_rois_square(X, bad_rois=[1, 6, 10], one_based=True)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y


def next_batch_cycler(loader, it_holder: dict):
    it = it_holder.get("it")
    if it is None:
        it = iter(loader)
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)
    it_holder["it"] = it
    return batch


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    conf = get_args()

    set_seed(conf.manual_seed)

    if conf.data == "ADNI":
        # ===== Load dataset 1 ADNI =====
        W_all, y_all = data_preprocess("./dataset/ADNI_processed.npz", conf.data)

    elif conf.data == "OASIS":
        # ===== Load dataset 2 OASIS =====
        W_all, y_all = data_preprocess("./dataset/OASIS_processed.npz", conf.data)

    # dataset = ConnectomeDataset(
    #     W_all, y_all, feature_fn=lambda W: compute_node_features(W)
    # )
    bin_edges, dataset = build_dataset(W_all, y_all)

    ################
    # data split
    n = len(dataset)
    n_train = int(n * conf.train_ratio)
    n_val = int(n * conf.val_ratio)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_labels = [data.y.item() for data in train_set]
    val_labels = [data.y.item() for data in val_set]
    test_labels = [data.y.item() for data in test_set]

    train_label_counts = Counter(train_labels)
    val_label_counts = Counter(val_labels)
    test_label_counts = Counter(test_labels)

    print(f"Train set label distribution: {train_label_counts}")
    print(f"Validation set label distribution: {val_label_counts}")
    print(f"Test set label distribution: {test_label_counts}")

    conf.train_loader = DataLoader(
        train_set,
        batch_size=conf.batch_size,
        shuffle=True,
        # collate_fn=collate_fn,
    )
    conf.val_loader = DataLoader(
        val_set,
        batch_size=conf.batch_size,
        shuffle=True,
        # collate_fn=collate_fn,
    )
    conf.test_loader = DataLoader(
        test_set,
        batch_size=conf.batch_size,
        shuffle=False,
        # collate_fn=collate_fn,
    )

    #### Trainer init
    if conf.optimizer == "FO-DSVRBGD":
        Trainer = Trainer_DecBiFirstOrder_pl_pruning_gnn(dataset, conf)
    elif conf.optimizer == "BrainNNExplainer":
        Trainer = Trainer_BrainNNExplainer(dataset, conf)
    elif conf.optimizer == "Basic_GNN":
        Trainer = Trainer_Basic_GNN(dataset, conf)

    Trainer.train()
