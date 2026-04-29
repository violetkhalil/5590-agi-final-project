from __future__ import annotations

import os, json
import gc
from decimal import Decimal
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from itertools import cycle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Subset
import math
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple
import time
import numpy as np
import torch
from Model.gnn import vec_to_symmetric_mask
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, Sampler
from Trainer.trainer_DecBilevelFirstOrder_pl_pruning_gnn import (
    Trainer_DecBiFirstOrder_pl_pruning_gnn,
)
from Trainer.trainer_BrainNNExplainer import Trainer_BrainNNExplainer
from Trainer.trainer_Basic_GNN import Trainer_Basic_GNN

from Trainer.trainer_DecBilevelFirstOrder_pl_pruning_gnn_CDR import (
    Trainer_DecBiFirstOrder_pl_pruning_gnn_CDR,
)
from Trainer.trainer_BrainNNExplainer_CDR import Trainer_BrainNNExplainer_CDR
from Trainer.trainer_Basic_GNN_CDR import Trainer_Basic_GNN_CDR
from Trainer.trainer_Basic_GNN_CDR_AUC import Trainer_Basic_GNN_CDR_AUC
from Trainer.trainer_BGAN_CDR import Trainer_BGAN_CDR
from Trainer.trainer_ALTER_CDR import Trainer_ALTER_CDR

from parameters import get_args
from utils.utils import *

# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from collections import Counter


class BalancedBatchSampler(Sampler):
    def __init__(self, pos_idx, neg_idx, batch_size, drop_last=False, seed=0):
        self.pos_idx = np.array(pos_idx, dtype=np.int64)
        self.neg_idx = np.array(neg_idx, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n = self.num_samples()
        num_batches = (
            n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        )
        for _ in range(num_batches):
            if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
                # fallback: sample from all indices
                all_idx = np.concatenate([self.pos_idx, self.neg_idx])
                batch = self.rng.choice(all_idx, size=self.batch_size, replace=True)
                yield batch.tolist()
                continue

            n_pos = max(1, self.batch_size // 2)
            n_neg = self.batch_size - n_pos
            pos = self.rng.choice(self.pos_idx, size=n_pos, replace=True)
            neg = self.rng.choice(self.neg_idx, size=n_neg, replace=True)
            batch = np.concatenate([pos, neg])
            self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return math.ceil(self.num_samples() / self.batch_size)

    def num_samples(self):
        return int(len(self.pos_idx) + len(self.neg_idx))


def print_label_stats_counter(split_name, labels, idx):
    y = labels[idx].tolist()
    cnt = Counter(y)
    total = len(y)
    print(f"[{split_name}] total={total}")
    for k in sorted(cnt.keys()):
        print(f"  class {k}: {cnt[k]:5d} ({cnt[k]/total*100:6.2f}%)")


def drop_rois_square(X, bad_rois, one_based=True):
    """
    从 (N, R, R) 的方阵堆栈中删除 bad_rois 对应的行和列。
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


def data_preprocess(file_path, dataset, task):
    data = np.load(file_path, allow_pickle=True)
    # for k in data.files:
    #     arr = data[k]
    #     if isinstance(arr, np.ndarray):
    #         print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
    #     else:
    #         print(f"{k}: {type(arr)}")

    X = data["normsc"]
    if dataset == "ADNI":
        X = drop_rois_square(X, bad_rois=[1, 6, 10], one_based=True)

    if task == "gender":
        y = data["sex"]

        le = LabelEncoder()
        y = le.fit_transform(y)

    elif task == "CDR":
        # For ADNI: CN, SMC, (EMCI, MCI, and LMCI)=MCI
        if dataset == "ADNI":
            y = data["group"]
            mapping = {"CN": 0, "SMC": 2, "EMCI": 1, "MCI": 1, "LMCI": 1}
            y = np.array([mapping[label] for label in y])
            print(f"✅ ADNI Label Aligned: {np.unique(y, return_counts=True)}")

        # For OASIS: AD(cdr>0.5) / MCI(cdr=0.5) / CN(cdr=0)
        elif dataset == "OASIS":
            y = data["cdr"]
            y_numeric = y.astype(float)
            labels = np.zeros(y_numeric.shape, dtype=int)

            labels[y_numeric == 0] = 0  # CN -> 0
            labels[y_numeric == 0.5] = 1  # MCI -> 1
            labels[y_numeric >= 1.0] = 2  # AD -> 2
            y = labels
            print(f"✅ OASIS Label Aligned: {np.unique(y, return_counts=True)}")

    return X, y.astype(np.int64)


def load_cdr_data(conf):
    # ===== Load dataset =====
    if conf.data == "ADNI":
        # ===== Load dataset 1 ADNI =====
        W_all, y_all = data_preprocess(
            "./dataset/ADNI_processed.npz", conf.data, conf.task
        )
    elif conf.data == "OASIS":
        # ===== Load dataset 2 OASIS =====
        W_all, y_all = data_preprocess(
            "./dataset/OASIS_processed.npz", conf.data, conf.task
        )
    else:
        raise ValueError(f"Unsupported dataset: {conf.data}")

    if conf.task == "CDR":
        if conf.cdr_pair == "default":
            keep = (y_all == 0) | (y_all == 1)
            y_all = y_all[keep]
            W_all = W_all[keep]
            conf.num_classes = 2
            print("[binary_test] keep labels {0,1}; set num_classes=2")
        else:
            if conf.data == "ADNI":
                if conf.cdr_pair == "NC_SMC":
                    remap = {0: 0, 2: 1}
                elif conf.cdr_pair == "SMC_MCI":
                    remap = {2: 0, 1: 1}
                else:
                    raise ValueError(
                        "ADNI cdr_pair must be NC_SMC or SMC_MCI, got: "
                        f"{conf.cdr_pair}"
                    )
            elif conf.data == "OASIS":
                if conf.cdr_pair == "NC_MCI":
                    remap = {0: 0, 1: 1}
                elif conf.cdr_pair == "MCI_AD":
                    remap = {1: 0, 2: 1}
                else:
                    raise ValueError(
                        "OASIS cdr_pair must be NC_MCI or MCI_AD, got: "
                        f"{conf.cdr_pair}"
                    )
            else:
                raise ValueError(f"Unsupported dataset for CDR pair: {conf.data}")

            keep = np.isin(y_all, list(remap.keys()))
            W_all = W_all[keep]
            y_all = np.vectorize(remap.get)(y_all[keep])
            conf.num_classes = 2
            print(
                f"[binary_test] cdr_pair={conf.cdr_pair}, remap={remap}; set num_classes=2"
            )

    return W_all, y_all


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
    
    W_all, y_all = load_cdr_data(conf)

    bin_edges, dataset = build_dataset(W_all, y_all)
    conf.N = W_all.shape[-1] 

    ################
    n = len(dataset)
    labels = np.array([dataset[i].y.item() for i in range(n)], dtype=np.int64)

    skf = StratifiedKFold(
        n_splits=conf.k_fold,
        shuffle=True,
        random_state=conf.manual_seed,
    )

    all_fold_metrics = []
    test_accs = []
    test_f1s = []
    test_aucs = []
    test_auprcs = []
    test_best_val_f1s = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), labels)):
        print(f"\n========== Fold {fold+1}/{conf.k_fold} ==========")

        set_seed(conf.manual_seed + fold)

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=conf.val_ratio,
            random_state=conf.manual_seed + fold,
        )
        y_train = labels[train_idx]
        inner_train_rel, inner_val_rel = next(
            sss.split(np.zeros(len(train_idx)), y_train)
        )

        inner_train_idx = train_idx[inner_train_rel]
        inner_val_idx = train_idx[inner_val_rel]

        train_set = Subset(dataset, inner_train_idx.tolist())
        val_set = Subset(dataset, inner_val_idx.tolist())
        test_set = Subset(dataset, test_idx.tolist())

        print_label_stats_counter("train", labels, inner_train_idx)
        print_label_stats_counter("val", labels, inner_val_idx)
        print_label_stats_counter("test", labels, test_idx)

        if conf.optimizer == "Basic_GNN_CDR_AUC" or conf.optimizer == "FO-DSVRBGD_CDR":
            y_train = labels[inner_train_idx]
            pos_rel = np.where(y_train == 1)[0]
            neg_rel = np.where(y_train == 0)[0]
            batch_sampler = BalancedBatchSampler(
                pos_rel, neg_rel, conf.batch_size, seed=conf.manual_seed + fold
            )
            conf.train_loader = DataLoader(train_set, batch_sampler=batch_sampler)
        else:
            conf.train_loader = DataLoader(
                train_set, batch_size=conf.batch_size, shuffle=True
            )

        if conf.optimizer == "FO-DSVRBGD_CDR":
            y_val = labels[inner_val_idx]
            pos_rel_val = np.where(y_val == 1)[0]
            neg_rel_val = np.where(y_val == 0)[0]
            val_sampler = BalancedBatchSampler(
                pos_rel_val, neg_rel_val, conf.batch_size, seed=conf.manual_seed + fold
            )
            conf.val_loader = DataLoader(val_set, batch_sampler=val_sampler)
        else:
            conf.val_loader = DataLoader(
                val_set, batch_size=conf.batch_size, shuffle=True
            )
        conf.test_loader = DataLoader(
            test_set, batch_size=conf.batch_size, shuffle=False
        )

        conf.fold = fold

        eta = float(Decimal(str(conf.epsilon)) ** 3)
        rho = float(Decimal(str(conf.epsilon)) * Decimal("0.2"))
        folder_name = (
            "epsilon-{}_eta-{}_rho-{}_momentum-{}_hidden_dim-{}_dropout-{}".format(
                conf.epsilon,
                eta,
                rho,
                conf.alpha_eta_product,
                conf.hidden_dim,
                conf.dropout,
            )
        )
        folder_name = os.path.join(folder_name, f"fold_{conf.fold}")
        conf.save_folder = os.path.join(
            "./checkpoints_pruning_pl_plus/gnn/{}/{}/{}/{}/".format(
                conf.task,
                conf.data,
                conf.cdr_pair,
                conf.optimizer,
            ),
            folder_name,
        )

        # ===== Trainer init =====
        # ===== 2 classes for gender =====
        if conf.optimizer == "FO-DSVRBGD":
            trainer = Trainer_DecBiFirstOrder_pl_pruning_gnn(dataset, conf)
        elif conf.optimizer == "BrainNNExplainer":
            trainer = Trainer_BrainNNExplainer(dataset, conf)
        elif conf.optimizer == "Basic_GNN":
            trainer = Trainer_Basic_GNN(dataset, conf)

        # ===== 2 classes for CDR =====
        elif conf.optimizer == "Basic_GNN_CDR":
            trainer = Trainer_Basic_GNN_CDR(dataset, conf)
        # elif conf.optimizer == "Basic_GNN_CDR_AUC":
            # trainer = Trainer_Basic_GNN_CDR_AUC(dataset, conf)
        elif conf.optimizer == "BrainNNExplainer_CDR":
            trainer = Trainer_BrainNNExplainer_CDR(dataset, conf)
        elif conf.optimizer == "BGAN_CDR":
            trainer = Trainer_BGAN_CDR(dataset, conf)

        elif conf.optimizer == "ALTER_CDR":
            conf.pos_encoding = "rrwp"
            trainer = Trainer_ALTER_CDR(dataset, conf)
        elif conf.optimizer == "BrainNETTF_CDR":
            conf.pos_encoding = "identity"
            trainer = Trainer_ALTER_CDR(dataset, conf)
        elif conf.optimizer == "FO-DSVRBGD_CDR":
            trainer = Trainer_DecBiFirstOrder_pl_pruning_gnn_CDR(dataset, conf)

        # ===== Train =====
        fold_result = trainer.train()
        all_fold_metrics.append(fold_result)

        # ===== Evaluate =====
        test_accs.append(float(fold_result["test_acc_at_best_val"]))
        test_f1s.append(float(fold_result["test_f1_at_best_val"]))
        test_aucs.append(float(fold_result["test_auc_at_best_val"]))
        test_auprcs.append(float(fold_result.get("test_auprc_at_best_val", -1.0)))
        test_best_val_f1s.append(float(fold_result.get("best_val_f1", -1.0)))

        print(
            f"[Fold {fold}] best_epoch={fold_result.get('best_epoch')}, "
            f"best_val_auprc={fold_result.get('best_val_auprc', -1.0):.4f}, "
            f"test_f1_at_best_val={fold_result.get('test_f1_at_best_val'):.4f}, "
            f"test_auc_at_best_val={fold_result.get('test_auc_at_best_val'):.4f}, "
            f"test_auprc_at_best_val={fold_result.get('test_auprc_at_best_val', -1.0):.4f}, "
            f"test_acc_at_best_val={fold_result.get('test_acc_at_best_val'):.4f}"
        )

        # clear GPU memory between folds
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_accs = np.array(test_accs, dtype=np.float64)
    mean_acc = float(test_accs.mean())
    std_acc = float(test_accs.std(ddof=1)) if len(test_accs) > 1 else 0.0

    test_aucs = np.array(test_aucs, dtype=np.float64)
    mean_auc = float(test_aucs.mean())
    std_auc = float(test_aucs.std(ddof=1)) if len(test_aucs) > 1 else 0.0

    test_auprcs = np.array(test_auprcs, dtype=np.float64)
    mean_auprc = float(test_auprcs.mean())
    std_auprc = float(test_auprcs.std(ddof=1)) if len(test_auprcs) > 1 else 0.0

    test_f1s = np.array(test_f1s, dtype=np.float64)
    mean_f1 = float(test_f1s.mean())
    std_f1 = float(test_f1s.std(ddof=1)) if len(test_f1s) > 1 else 0.0

    cms = []
    for d in all_fold_metrics:
        if "CM" in d:
            cm = np.array(d["CM"], dtype=np.float64)
            cms.append(cm)

    print(f"\n==== {conf.k_fold}-fold CV Result ====")
    print(f"Test F1-score (at best val): mean={mean_f1:.4f}, std={std_f1:.4f}")
    print(f"Test Auc (at best val): mean={mean_auc:.4f}, std={std_auc:.4f}")
    print(f"Test AUPRC (at best val): mean={mean_auprc:.4f}, std={std_auprc:.4f}")
    print(f"Test Acc (at best val): mean={mean_acc:.4f}, std={std_acc:.4f}")

    sys.stdout.flush()

    first_fold_dir = all_fold_metrics[0]["save_folder"]
    cv_save_path = os.path.join(os.path.dirname(first_fold_dir), "cv_results.txt")

    mean_cm = None
    if len(cms) > 0:
        mean_cm = np.mean(np.stack(cms, axis=0), axis=0)
        cm_dir = os.path.dirname(first_fold_dir)
        np.save(os.path.join(cm_dir, "cm_mean.npy"), mean_cm)
        np.savetxt(os.path.join(cm_dir, "cm_mean.csv"), mean_cm, delimiter=",")
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            mean_cm,
            cmap="Blues",
            annot=True,
            fmt=".2f",
            cbar=True,
            square=True,
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Mean Confusion Matrix")
        cm_pdf = os.path.join(cm_dir, "cm_mean.pdf")
        plt.tight_layout()
        plt.savefig(cm_pdf, dpi=500, bbox_inches="tight")
        plt.close()
        print("[cm] save mean CM figure to:", cm_dir)
        print("[cm] save mean CM to:", cm_dir)

    if ("FO-DSVRBGD" in conf.optimizer) or ("BrainNNExplainer" in conf.optimizer):
        N = conf.N
        M_np = np.zeros((N, N), dtype=np.float64)
        for fold_id in range(conf.k_fold):
            ckpt_path = os.path.join(
                all_fold_metrics[fold_id]["save_folder"], "backbone_mask.pt"
            )
            ckpt = torch.load(ckpt_path, map_location="cpu")
            mask_vec = ckpt["mask"].float().view(-1)

            if "FO-DSVRBGD" in conf.optimizer:
                M = torch.sigmoid(mask_vec).reshape(N, N)
                M.fill_diagonal_(0.0)

            elif "BrainNNExplainer" in conf.optimizer:
                M = vec_to_symmetric_mask(torch.sigmoid(mask_vec), N)

            M_np += M.detach().cpu().numpy()

        M_np /= float(conf.k_fold)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            M_np,
            cmap="viridis",
            square=True,
            cbar=True,
            vmin=0.0,
            vmax=1.0,
        )
        plt.xlabel("Node index")
        plt.ylabel("Node index")
        plt.title("Learned Shared Edge Mask (probabilities)")

        out_path = os.path.join(os.path.dirname(first_fold_dir), "mask_matrix.pdf")
        plt.tight_layout()
        plt.savefig(out_path, dpi=500, bbox_inches="tight")
        plt.close()
        print("[plot_mask] save mask figure =", out_path)

    with open(cv_save_path, "w", encoding="utf-8") as f:
        summary = {
            "k_fold": int(conf.k_fold),
            "seed": int(conf.manual_seed),
            "data": conf.data,
            "optimizer": conf.optimizer,
            "metric": "test_auprc_at_best_val",
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "mean_auprc": mean_auprc,
            "std_auprc": std_auprc,
            "folds": int(len(test_accs)),
        }
        if mean_cm is not None:
            summary["mean_cm"] = mean_cm.tolist()
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")

        for d in all_fold_metrics:
            f.write(json.dumps({"fold_result": d}, ensure_ascii=False) + "\n")
    print(f"Saved fold metrics to: {cv_save_path}")
