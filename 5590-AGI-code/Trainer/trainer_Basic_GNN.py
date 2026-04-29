import shutil
import os
from datetime import datetime
import torch
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
import time
import torch.nn.functional as F
import sys
import torch.distributed as dist
import pickle
from decimal import Decimal
import torch.nn as nn
from Model.gnn import *
from utils.utils import *


class Trainer_Basic_GNN:
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

        #### Log
        folder_name = "epsilon-{}_eta-{}_rho-{}_momentum-{}_hidden_dim-{}_dropout-{}".format(
            self.conf.epsilon,
            self.eta,
            self.rho,
            self.conf.alpha_eta_product,
            self.conf.hidden_dim,
            self.conf.dropout,
        )

        folder_name = os.path.join(folder_name, f"fold_{self.conf.fold}")

        save_folder = os.path.join(
            "./checkpoints_pruning_pl/gnn/{}/{}/{}/".format(
                self.conf.task,
                self.conf.data,
                self.conf.optimizer,
            ),
            folder_name,
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

        def classification_loss(logits, y):
            return F.cross_entropy(logits, y.long())

        def evaluate(loader):
            backbone_model.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logits = backbone_model(batch)
                    loss = classification_loss(logits, batch.y)

                    pred = logits.argmax(dim=-1)
                    total_loss += loss.item() * batch.num_graphs
                    total_correct += (pred == batch.y.long()).sum().item()
                    total_samples += batch.num_graphs

            backbone_model.train()
            avg_loss = total_loss / total_samples
            acc = total_correct / total_samples
            return avg_loss, acc

        train_loss, train_acc = evaluate(self.conf.train_loader)
        val_loss, val_acc = evaluate(self.conf.val_loader)
        test_loss, test_acc = evaluate(self.conf.test_loader)

        self.result["train_loss"][0] = train_loss
        self.result["val_loss"][0] = val_loss
        self.result["val_acc"][0] = val_acc
        self.result["test_loss"][0] = test_loss
        self.result["test_acc"][0] = test_acc
        self.result["total_time"][0] = 0.0

        outer_steps = self.conf.num_epochs
        total_time = 0.0
        best_val_acc = -1.0
        best_epoch = -1
        best_test_acc_at_best_val = -1.0

        for epoch in range(1, outer_steps + 1):
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

            val_loss, val_acc = evaluate(self.conf.val_loader)
            test_loss, test_acc = evaluate(self.conf.test_loader)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_test_acc_at_best_val = test_acc
                best_epoch = epoch
                torch.save({"backbone_state_dict": backbone_model.state_dict()}, params_path)
                print("save params:", params_path)

            total_time += time.time() - epoch_start

            self.result["train_loss"][epoch] = train_loss
            self.result["train_acc"][epoch] = train_acc
            self.result["val_loss"][epoch] = val_loss
            self.result["val_acc"][epoch] = val_acc
            self.result["test_loss"][epoch] = test_loss
            self.result["test_acc"][epoch] = test_acc
            self.result["total_time"][epoch] = total_time

            print(
                f"epoch={epoch}, time={total_time:.3f}, "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
            )
            sys.stdout.flush()

        save_fig(outer_steps, self.result, save_folder, best_test_acc_at_best_val)

        return {
            "fold": getattr(self.conf, "fold", None),
            "best_epoch": best_epoch,
            "best_val_acc": float(best_val_acc),
            "test_acc_at_best_val": float(best_test_acc_at_best_val),
            "final_test_acc": float(self.result["test_acc"][outer_steps]),
            "save_folder": save_folder,
            "params_path": params_path,
        }
