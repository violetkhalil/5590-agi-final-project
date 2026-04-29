from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import shutil
from datetime import datetime
import torch
import numpy as np
import copy
import time
import torch.nn.functional as F
import sys
import os
import torch.distributed as dist
import pickle
from decimal import Decimal
import torch.nn as nn
from Model.gnn import *


class Trainer_BrainNNExplainer_CDR:
    def __init__(self, dataset, conf):
        self.dataset = dataset
        self.conf = conf
        self.eval_interval = 1
        self.eta = float(Decimal(str(self.conf.epsilon)) ** 3)
        self.rho = float(Decimal(str(self.conf.epsilon)) * Decimal("0.2"))
        self.alpha = self.conf.alpha_eta_product / (self.eta**2)

        self.result = {}
        k_list = [
            "local_train_loss",
            "full_train_loss",
            "local_val_loss",
            "full_val_loss",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "test_loss",
            "test_acc",
            "train_auc",
            "train_f1",
            "val_auc",
            "val_f1",
            "test_auc",
            "test_f1",
            "train_auprc",
            "val_auprc",
            "test_auprc",
            "total_n_bits",
            "total_time",
        ]
        for k in k_list:
            self.result[k] = np.zeros(self.conf.num_epochs + 1)

    def local_logistic_loss(self, x_train_batch, y_train_batch):
        def f(w):
            return F.binary_cross_entropy_with_logits(x_train_batch @ w, y_train_batch)

        return f

    def train(self):
        torch.set_default_dtype(torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        N = self.conf.N

        #### Log (save path passed from main)
        save_folder = getattr(self.conf, "save_folder", None)
        if save_folder is None:
            raise ValueError(
                "conf.save_folder is not set. Please set it in main before training."
            )

        if os.path.isdir(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

        file_addr = os.path.join(save_folder, "result.pkl")

        params_path = os.path.join(save_folder, "backbone_mask.pt")

        #########################

        num_features = self.dataset[0].x.shape[-1]
        args = self.conf

        backbone_model = IBGNN(
            IBGConv(num_features, args, num_classes=self.conf.num_classes),
            mlp=None,
            pooling=args.pooling,
        ).to(device)

        opt_backbone = torch.optim.Adam(
            backbone_model.parameters(),
            lr=self.eta,
            weight_decay=0.00001,
        )

        len_mask = N * (N - 1) // 2
        # mask_vec = nn.Parameter(torch.randn(len_mask, device=device))
        mask_vec = nn.Parameter(torch.ones(len_mask, device=device))
        opt_mask = torch.optim.Adam([mask_vec], lr=0.01)

        def classification_loss(logits, y):
            return F.cross_entropy(logits, y.long())

        def evaluate(loader, use_mask=False):
            was_training = backbone_model.training
            backbone_model.eval()
            total_loss = 0.0
            total_samples = 0

            all_y_true = []
            all_y_scores = []
            all_y_preds = []

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)

                    if use_mask:
                        pruned_edge_mask = prune_edge_mask(
                            vec_to_symmetric_mask(mask_vec, self.conf.N), batch.edge_flag
                        ).to(device)
                        batch.edge_attr = (
                            batch.edge_attr * pruned_edge_mask.view(1, -1).sigmoid()
                        ).squeeze()
                        batch.edge_flag = pruned_edge_mask

                    logits = backbone_model(batch)
                    loss = classification_loss(logits, batch.y)

                    probs = F.softmax(logits, dim=-1)
                    preds = logits.argmax(dim=-1)

                    total_loss += loss.item() * batch.num_graphs
                    total_samples += batch.num_graphs

                    all_y_true.append(batch.y.cpu().numpy())
                    all_y_preds.append(preds.cpu().numpy())
                    all_y_scores.append(probs.cpu().numpy())

            if was_training:
                backbone_model.train()

            y_true = np.concatenate(all_y_true)
            y_preds = np.concatenate(all_y_preds)
            y_scores = np.concatenate(all_y_scores)

            avg_loss = total_loss / max(total_samples, 1)
            acc = (y_preds == y_true).mean()
            f1 = f1_score(y_true, y_preds, average="macro")

            try:
                if self.conf.num_classes == 2:
                    auc = roc_auc_score(y_true, y_scores[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_scores, multi_class="ovr")
            except:
                auc = 0.5

            try:
                if self.conf.num_classes == 2:
                    auprc = average_precision_score(y_true, y_scores[:, 1])
                else:
                    y_true_bin = label_binarize(
                        y_true, classes=list(range(self.conf.num_classes))
                    )
                    auprc = average_precision_score(
                        y_true_bin, y_scores, average="macro"
                    )
            except ValueError:
                auprc = 0.0

            cm = confusion_matrix(y_true, y_preds)
            f1_each_class = f1_score(y_true, y_preds, average=None)

            return avg_loss, acc, f1, auc, auprc, cm, f1_each_class

        train_loss, train_acc, train_f1, train_auc, train_auprc, _, _ = evaluate(
            self.conf.train_loader
        )
        val_loss, val_acc, val_f1, val_auc, val_auprc, _, _ = evaluate(
            self.conf.val_loader
        )
        (
            test_loss,
            test_acc,
            test_f1,
            test_auc,
            test_auprc,
            test_cm,
            test_f1_each_class,
        ) = evaluate(self.conf.test_loader)

        self.result["train_loss"][0] = train_loss
        self.result["val_loss"][0] = val_loss
        self.result["val_acc"][0] = val_acc
        self.result["test_loss"][0] = test_loss
        self.result["test_acc"][0] = test_acc
        self.result["val_f1"][0] = val_f1
        self.result["val_auc"][0] = val_auc
        self.result["test_f1"][0] = test_f1
        self.result["test_auc"][0] = test_auc
        self.result["train_auprc"][0] = train_auprc
        self.result["val_auprc"][0] = val_auprc
        self.result["test_auprc"][0] = test_auprc
        self.result["total_time"][0] = 0.0

        total_epochs = self.conf.num_epochs
        phase1_epochs = int(0.6 * total_epochs)
        phase2_epochs = int(0.2 * total_epochs)
        phase3_epochs = total_epochs - phase1_epochs - phase2_epochs

        total_time = 0

        best_val_f1 = -1.0
        best_val_loss = float("inf")
        best_epoch = -1
        best_test_f1_at_best_val = -1.0
        best_test_acc_at_best_val = -1.0
        best_test_auc_at_best_val = -1.0
        best_test_auprc_at_best_val = -1.0

        best_test_cm = test_cm
        best_test_f1_each_class = test_f1_each_class

        for epoch in range(1, phase1_epochs + 1):
            epoch_start = time.time()

            backbone_model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_samples = 0

            for batch in self.conf.train_loader:
                batch = batch.to(device)

                opt_backbone.zero_grad()
                logits = backbone_model(batch)
                loss = classification_loss(logits, batch.y)
                loss.backward()
                opt_backbone.step()

                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    train_loss_sum += loss.item() * batch.num_graphs
                    train_correct += (pred == batch.y.long()).sum().item()
                    train_samples += batch.num_graphs

            train_loss = train_loss_sum / max(train_samples, 1)
            train_acc = train_correct / max(train_samples, 1)

            # eval on original graph
            val_loss, val_acc, val_f1, val_auc, val_auprc, _, _ = evaluate(
                self.conf.val_loader, use_mask=False
            )
            test_loss, test_acc, test_f1, test_auc, test_auprc, _, _ = evaluate(
                self.conf.test_loader, use_mask=False
            )

            total_time += time.time() - epoch_start

            self.result["train_loss"][epoch] = train_loss
            self.result["train_acc"][epoch] = train_acc
            self.result["val_loss"][epoch] = val_loss
            self.result["val_acc"][epoch] = val_acc
            self.result["val_f1"][epoch] = val_f1
            self.result["val_auc"][epoch] = val_auc
            self.result["val_auprc"][epoch] = val_auprc
            self.result["test_loss"][epoch] = test_loss
            self.result["test_acc"][epoch] = test_acc
            self.result["test_f1"][epoch] = test_f1
            self.result["test_auc"][epoch] = test_auc
            self.result["test_auprc"][epoch] = test_auprc

            print(
                f"epoch={epoch}, phase=1, time={total_time:.3f}\n"
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}\n"
                f"F1 Score: Val={val_f1:.4f}, Test={test_f1:.4f} | "
                f"AUC: Val={val_auc:.4f}, Test={test_auc:.4f} | "
                f"AUPRC: Val={val_auprc:.4f}, Test={test_auprc:.4f}"
            )
            sys.stdout.flush()

        for p in backbone_model.parameters():
            p.requires_grad_(False)

        backbone_model.eval()

        for epoch in range(phase1_epochs + 1, phase1_epochs + phase2_epochs + 1):
            epoch_start = time.time()

            train_loss_sum = 0.0
            train_correct = 0
            train_samples = 0

            for batch in self.conf.train_loader:
                batch = batch.to(device)

                with torch.no_grad():
                    logits_orig = backbone_model(batch)
                    loss_p = classification_loss(logits_orig, batch.y)

                # 2) masked loss (grad on mask_vec)
                opt_mask.zero_grad()

                ############### refer original code
                pruned_edge_mask = prune_edge_mask(
                    vec_to_symmetric_mask(mask_vec, N), batch.edge_flag
                ).to(device)
                explained_edge_attr = batch.edge_attr * pruned_edge_mask.view(1, -1).sigmoid()
                explained_edge_attr = explained_edge_attr.squeeze()
                batch.edge_attr = explained_edge_attr
                batch.edge_flag = pruned_edge_mask

                logits_masked = backbone_model(batch)
                loss_m = classification_loss(logits_masked, batch.y)

                mask_sig = torch.sigmoid(mask_vec)
                sparsity = mask_sig.mean()
                eps = 1e-8
                entropy = -(
                    mask_sig * torch.log(mask_sig + eps)
                    + (1.0 - mask_sig) * torch.log(1.0 - mask_sig + eps)
                ).mean()

                loss = loss_p + loss_m + sparsity + entropy
                loss.backward()
                opt_mask.step()

                with torch.no_grad():
                    pred = logits_masked.argmax(dim=-1)
                    train_loss_sum += loss.item() * batch.num_graphs
                    train_correct += (pred == batch.y.long()).sum().item()
                    train_samples += batch.num_graphs

            train_loss = train_loss_sum / max(train_samples, 1)
            train_acc = train_correct / max(train_samples, 1)

            val_loss, val_acc, val_f1, val_auc, val_auprc, _, _ = evaluate(
                self.conf.val_loader, use_mask=True
            )
            (
                test_loss,
                test_acc,
                test_f1,
                test_auc,
                test_auprc,
                best_test_cm,
                best_test_f1_each_class,
            ) = evaluate(self.conf.test_loader, use_mask=True)

            total_time += time.time() - epoch_start

            self.result["train_loss"][epoch] = train_loss
            self.result["train_acc"][epoch] = train_acc
            self.result["val_loss"][epoch] = val_loss
            self.result["val_acc"][epoch] = val_acc
            self.result["test_loss"][epoch] = test_loss
            self.result["test_acc"][epoch] = test_acc
            self.result["total_time"][epoch] = total_time
            self.result["val_f1"][epoch] = val_f1
            self.result["val_auc"][epoch] = val_auc
            self.result["val_auprc"][epoch] = val_auprc
            self.result["test_f1"][epoch] = test_f1
            self.result["test_auc"][epoch] = test_auc
            self.result["test_auprc"][epoch] = test_auprc

            print(
                f"epoch={epoch}, phase=2, time={total_time:.3f}\n"
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}\n"
                f"F1 Score: Val={val_f1:.4f}, Test={test_f1:.4f} | "
                f"AUC: Val={val_auc:.4f}, Test={test_auc:.4f} | "
                f"AUPRC: Val={val_auprc:.4f}, Test={test_auprc:.4f}"
            )
            sys.stdout.flush()

        for p in backbone_model.parameters():
            p.requires_grad_(True)
        mask_vec.requires_grad_(False)

        backbone_model.train()

        for epoch in range(phase1_epochs + phase2_epochs + 1, total_epochs + 1):
            epoch_start = time.time()

            train_loss_sum = 0.0
            train_correct = 0
            train_samples = 0

            for batch in self.conf.train_loader:
                batch = batch.to(device)

                ############### refer original code
                pruned_edge_mask = prune_edge_mask(
                    vec_to_symmetric_mask(mask_vec, N), batch.edge_flag
                ).to(device)
                explained_edge_attr = batch.edge_attr * pruned_edge_mask.view(1, -1).sigmoid()
                explained_edge_attr = explained_edge_attr.squeeze()
                batch.edge_attr = explained_edge_attr
                batch.edge_flag = pruned_edge_mask

                opt_backbone.zero_grad()
                logits = backbone_model(batch)
                loss = classification_loss(logits, batch.y)
                loss.backward()
                opt_backbone.step()

                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    train_loss_sum += loss.item() * batch.num_graphs
                    train_correct += (pred == batch.y.long()).sum().item()
                    train_samples += batch.num_graphs

            train_loss = train_loss_sum / max(train_samples, 1)
            train_acc = train_correct / max(train_samples, 1)

            val_loss, val_acc, val_f1, val_auc, val_auprc, _, _ = evaluate(
                self.conf.val_loader, use_mask=True
            )
            (
                test_loss,
                test_acc,
                test_f1,
                test_auc,
                test_auprc,
                test_cm,
                test_f1_each_class,
            ) = evaluate(self.conf.test_loader, use_mask=True)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_f1
                best_epoch = epoch
                best_test_f1_at_best_val = test_f1
                best_test_acc_at_best_val = test_acc
                best_test_auc_at_best_val = test_auc
                best_test_auprc_at_best_val = test_auprc
                best_test_cm = test_cm
                best_test_f1_each_class = test_f1_each_class
                torch.save(
                    {
                        "backbone_state_dict": backbone_model.state_dict(),
                        "mask": mask_vec.detach().cpu(),
                    },
                    params_path,
                )
                print(f"--- Save Best Model (Val Loss: {val_loss:.4f}) at epoch {epoch} ---")

            total_time += time.time() - epoch_start

            self.result["train_loss"][epoch] = train_loss
            self.result["train_acc"][epoch] = train_acc
            self.result["val_loss"][epoch] = val_loss
            self.result["val_acc"][epoch] = val_acc
            self.result["val_f1"][epoch] = val_f1
            self.result["val_auc"][epoch] = val_auc
            self.result["val_auprc"][epoch] = val_auprc
            self.result["test_loss"][epoch] = test_loss
            self.result["test_acc"][epoch] = test_acc
            self.result["test_f1"][epoch] = test_f1
            self.result["test_auc"][epoch] = test_auc
            self.result["test_auprc"][epoch] = test_auprc

            print(
                f"epoch={epoch}, phase=3, time={total_time:.3f}\n"
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}\n"
                f"F1 Score: Val={val_f1:.4f}, Test={test_f1:.4f} | "
                f"AUC: Val={val_auc:.4f}, Test={test_auc:.4f} | "
                f"AUPRC: Val={val_auprc:.4f}, Test={test_auprc:.4f}"
            )
            sys.stdout.flush()

        print("total time = {}".format(total_time))

        with open(file_addr, "wb") as f:
            pickle.dump(self.result, f)
        print("save file = ", file_addr)

        plot_mask_half(save_folder, N)

        save_fig_cdr(
            total_epochs,
            self.result,
            save_folder,
            best_test_f1_at_best_val,
            best_test_acc_at_best_val,
            best_test_auc_at_best_val,
            best_test_cm,
            best_test_f1_each_class,
        )

        return {
            "fold": getattr(self.conf, "fold", None),
            "best_epoch": best_epoch,
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "test_f1_at_best_val": float(best_test_f1_at_best_val),
            "test_acc_at_best_val": float(best_test_acc_at_best_val),
            "test_auc_at_best_val": float(best_test_auc_at_best_val),
            "test_auprc_at_best_val": float(best_test_auprc_at_best_val),
            "final_test_f1": float(self.result["test_f1"][total_epochs]),
            "save_folder": save_folder,
            "params_path": params_path,
            "CM": best_test_cm.tolist(),
        }
