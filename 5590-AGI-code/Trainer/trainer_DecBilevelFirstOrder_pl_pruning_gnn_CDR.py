from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
import shutil
import matplotlib.pyplot as plt
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


def _best_f1_threshold(y_true, y_score_pos):
    """Scan candidate thresholds; return the one maximizing macro-F1 on (y_true, y_score_pos)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    cands = np.unique(y_score_pos)
    if len(cands) > 200:
        cands = np.quantile(y_score_pos, np.linspace(0.01, 0.99, 200))
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        y_pred = (y_score_pos >= t).astype(np.int64)
        f1 = f1_score(y_true, y_pred, average="macro")
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


class Trainer_DecBiFirstOrder_pl_pruning_gnn_CDR:
    def __init__(self, dataset, conf):
        self.dataset = dataset
        self.conf = conf
        self.eval_interval = 1
        self.eta = float(Decimal(str(self.conf.epsilon)) ** 3)

        self.eta_x_expand = float(self.conf.eta_x_expand)

        # self.rho = float(Decimal(str(self.conf.epsilon)) * Decimal("0.2"))
        # self.rho = float(
        #     Decimal(str(self.conf.epsilon)) * Decimal(str(self.conf.rho_coeff))
        # )
        self.rho = float(Decimal(str(self.conf.rho)))

        # self.alpha = self.conf.alpha_eta_product / (self.eta**2)
        self.alpha = self.conf.alpha_eta_product / (self.eta)

        self.result = {}
        k_list = [
            "local_train_loss",
            "full_train_loss",
            "local_val_loss",
            "full_val_loss",
            "upper_loss",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "test_loss",
            "test_acc",
            "train_auc",
            "train_auprc",
            "train_f1",
            "val_auc",
            "val_auprc",
            "val_f1",
            "test_auc",
            "test_auprc",
            "test_f1",
            "total_n_bits",
            "total_time",
        ]
        for k in k_list:
            self.result[k] = np.zeros(self.conf.num_epochs + 1)

    def train(self):
        def classification_loss(logits, y):
            return F.cross_entropy(logits, y.long())

        def evaluate(loader, threshold=None):
            backbone_model.eval()
            total_loss = 0.0
            total_samples = 0

            all_y_true = []
            all_y_scores = []
            all_y_preds = []

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logits = backbone_model(batch)
                    loss = classification_loss(logits, batch.y)

                    probs = F.softmax(logits, dim=-1)
                    preds = logits.argmax(dim=-1)

                    total_loss += loss.item() * batch.num_graphs
                    total_samples += batch.num_graphs

                    all_y_true.append(batch.y.cpu().numpy())
                    all_y_preds.append(preds.cpu().numpy())
                    all_y_scores.append(probs.cpu().numpy())

            backbone_model.train()

            y_true = np.concatenate(all_y_true)
            y_scores = np.concatenate(all_y_scores)

            if threshold is not None and y_scores.shape[1] == 2:
                y_preds = (y_scores[:, 1] >= threshold).astype(np.int64)
            else:
                y_preds = np.concatenate(all_y_preds)

            avg_loss = total_loss / total_samples
            acc = (y_preds == y_true).mean()

            f1 = f1_score(y_true, y_preds, average="macro")

            try:
                if self.conf.num_classes == 2:
                    auc = roc_auc_score(y_true, y_scores[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_scores, multi_class="ovr")
            except ValueError:
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

            return avg_loss, acc, f1, auc, auprc, y_true, y_scores

        torch.set_default_dtype(torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_val_acc = -1.0
        best_val_f1 = -1.0
        best_val_auprc = -1.0
        best_epoch = -1
        best_test_acc_at_best_val = -1.0
        best_test_f1_at_best_val = -1.0
        best_test_auc_at_best_val = -1.0
        best_test_auprc_at_best_val = -1.0

        N = self.conf.N

        num_epochs = self.conf.num_epochs
        pre_epochs = num_epochs // 2

        save_folder = getattr(self.conf, "save_folder", None)
        if save_folder is None:
            raise ValueError(
                "conf.save_folder is not set. Please set it in main before training."
            )

        if os.path.isdir(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

        pretrain_path = os.path.join(save_folder, "pretrained_backbone.pt")
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
            lr=0.001,
            weight_decay=0.0001,
        )
        sched_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_backbone, T_max=pre_epochs
        )

        total_time = 0

        for epoch in range(1, pre_epochs + 1):
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

            val_loss, val_acc, val_f1, val_auc, val_auprc, y_true_val, y_score_val = evaluate(
                self.conf.val_loader
            )
            tau = _best_f1_threshold(y_true_val, y_score_val[:, 1]) if y_score_val.shape[1] == 2 else None
            test_loss, test_acc, test_f1, test_auc, test_auprc, _, _ = evaluate(
                self.conf.test_loader, threshold=tau
            )

            if val_auprc >= best_val_auprc:
                best_val_f1 = val_f1
                best_val_auprc = val_auprc
                best_test_f1_at_best_val = test_f1
                best_test_acc_at_best_val = test_acc
                best_test_auc_at_best_val = test_auc
                best_test_auprc_at_best_val = test_auprc
                best_epoch = epoch
                torch.save(
                    {"pretrained_backbone_state_dict": backbone_model.state_dict()}, pretrain_path
                )
                print("save params:", pretrain_path)

            total_time += time.time() - epoch_start

            self.result["train_loss"][epoch] = train_loss
            self.result["upper_loss"][epoch] = val_loss
            self.result["val_acc"][epoch] = val_acc
            self.result["val_f1"][epoch] = val_f1
            self.result["val_auc"][epoch] = val_auc
            self.result["val_auprc"][epoch] = val_auprc
            self.result["test_loss"][epoch] = test_loss
            self.result["test_acc"][epoch] = test_acc
            self.result["test_f1"][epoch] = test_f1
            self.result["test_auc"][epoch] = test_auc
            self.result["test_auprc"][epoch] = test_auprc
            self.result["total_time"][epoch] = total_time

            print(
                f"epoch={epoch}, time={total_time:.3f}\n"
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}\n"
                f"F1 Score: Val={val_f1:.4f}, Test={test_f1:.4f} | "
                f"AUC: Val={val_auc:.4f}, Test={test_auc:.4f} | "
                f"AUPRC: Val={val_auprc:.4f}, Test={test_auprc:.4f}"
            )
            sys.stdout.flush()

            sched_backbone.step()

        ckpt = torch.load(pretrain_path, map_location=device)
        backbone_model.load_state_dict(ckpt["pretrained_backbone_state_dict"])

        ################## load model
        # pretrain_weight_path = "./checkpoints_pruning_pl/gnn/{}/Basic_GNN/epsilon-0.08_eta-0.000512_rho-0.016_momentum-0.9_hidden_dim-16_dropout-0.3/fold_{}/backbone_mask.pt".format(
        #     self.conf.data, self.conf.fold
        # )
        # ckpt = torch.load(pretrain_weight_path, map_location=device)
        # backbone_model.load_state_dict(ckpt["backbone_state_dict"])
        # print(pretrain_weight_path)

        # ============ params=======
        # —— y —— #
        spec_all = build_param_spec_all(backbone_model)
        params_vec0 = pack_params_from_model(backbone_model, spec_all).to(device)

        sl_all, len_all = flat_slices_from_spec(spec_all)

        # len_mask = N * (N - 1) // 2
        # hparams_vec0 = torch.randn(len_mask, device=device)
        # hparams_vec0 = torch.zeros(len_mask, device=device)
        # hparams_vec0 = torch.ones(len_mask, device=device)

        hparams_vec0 = torch.zeros(N * N, device=device)
        hparams_vec0.requires_grad_(True)

        z_vec0 = params_vec0.clone().detach().requires_grad_(True)

        hparams = [hparams_vec0.clone().detach().requires_grad_(True)]
        params = [params_vec0.clone().detach().requires_grad_(True)]
        z = z_vec0.clone().detach().requires_grad_(True)

        u_1 = torch.zeros_like(hparams[0])
        u_2 = torch.zeros_like(hparams[0])
        u_3 = torch.zeros_like(hparams[0])
        v_1 = torch.zeros_like(params[0])
        v_2 = torch.zeros_like(params[0])
        w_1 = torch.zeros_like(z)

        # ============ evaluation =======
        train_loss, train_acc, train_f1, train_auc, train_auprc, _, _, _, _ = eval_acc_gnn(
            self.conf.train_loader,
            backbone_model,
            spec_all,
            params[0].detach().clone(),
            hparams[0].detach().clone(),
            N,
        )
        val_loss, val_acc, val_f1, val_auc, val_auprc, _, _, y_true_val, y_score_val = eval_acc_gnn(
            self.conf.val_loader,
            backbone_model,
            spec_all,
            params[0].detach().clone(),
            hparams[0].detach().clone(),
            N,
        )
        tau = _best_f1_threshold(y_true_val, y_score_val[:, 1]) if y_score_val.shape[1] == 2 else None
        (
            test_loss,
            test_acc,
            test_f1,
            test_auc,
            test_auprc,
            best_test_cm,
            best_test_f1_each_class,
            _,
            _,
        ) = eval_acc_gnn(
            self.conf.test_loader,
            backbone_model,
            spec_all,
            params[0].detach().clone(),
            hparams[0].detach().clone(),
            N,
            threshold=tau,
        )

        self.result["train_loss"][0] = train_loss
        self.result["train_auprc"][0] = train_auprc
        self.result["upper_loss"][0] = val_loss
        self.result["val_acc"][0] = val_acc
        self.result["val_f1"][0] = val_f1
        self.result["val_auc"][0] = val_auc
        self.result["val_auprc"][0] = val_auprc
        self.result["test_loss"][0] = test_loss
        self.result["test_acc"][0] = test_acc
        self.result["test_f1"][0] = test_f1
        self.result["test_auc"][0] = test_auc
        self.result["test_auprc"][0] = test_auprc

        def next_batch(loader, it):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            return batch, it

        global_step = 0
        best_val_acc = 0
        best_val_f1 = -1
        best_val_auprc = -1

        soap_loss = SOAPLOSS(
            threshold=0.5,
            data_length=len(self.dataset),
            loss_type="sqh",
            device=device,
        )

        print("\nPretain Finish! Begin ours method!\n")

        for epoch in range(pre_epochs + 1, num_epochs + 1):
            epoch_train_loss_sum = 0.0
            epoch_train_steps = 0

            steps_per_epoch = len(self.conf.train_loader)

            train_iter = iter(self.conf.train_loader)
            val_iter = iter(self.conf.val_loader)

            for step_in_epoch in range(steps_per_epoch):
                step_start_time = time.time()

                # ============update x ==============
                batch_val, val_iter = next_batch(self.conf.val_loader, val_iter)
                loss = upper_loss_gnn_CDR(
                    batch_val,
                    backbone_model,
                    spec_all,
                    params[0],
                    hparams[0],
                    N,
                    soap_loss,
                )
                fy_x = torch.autograd.grad(loss, hparams, create_graph=False)[0]
                del loss; torch.cuda.empty_cache()

                batch_train1, train_iter = next_batch(self.conf.train_loader, train_iter)
                loss = lower_loss_gnn_CDR(
                    batch_train1,
                    backbone_model,
                    spec_all,
                    params[0],
                    hparams[0],
                    N,
                    soap_loss,
                )
                gy_x = torch.autograd.grad(loss, hparams, create_graph=False)[0]
                del loss; torch.cuda.empty_cache()

                batch_train2, train_iter = next_batch(self.conf.train_loader, train_iter)
                loss = lower_loss_gnn_CDR(
                    batch_train2,
                    backbone_model,
                    spec_all,
                    z,
                    hparams[0],
                    N,
                    soap_loss,
                )
                gz_x = torch.autograd.grad(loss, hparams, create_graph=False)[0]
                del loss; torch.cuda.empty_cache()

                if global_step == 0:
                    u_1.data = torch.clone(fy_x).detach()
                    u_2.data = torch.clone(gy_x).detach()
                    u_3.data = torch.clone(gz_x).detach()
                else:
                    u_1.data = (1 - self.alpha * self.eta) * (u_1.data) + (
                        self.alpha * self.eta
                    ) * fy_x.data
                    u_2.data = (1 - self.alpha * self.eta) * (u_2.data) + (
                        self.alpha * self.eta
                    ) * gy_x.data
                    u_3.data = (1 - self.alpha * self.eta) * (u_3.data) + (
                        self.alpha * self.eta
                    ) * gz_x.data
                m_x = u_1.data + (1 / self.rho) * (u_2.data - u_3.data)
                # check_stats("u_1", u_1)
                # check_stats("u_2", u_2)
                # check_stats("u_3", u_3)

                # =========update y=========
                batch_val_y, val_iter = next_batch(self.conf.val_loader, val_iter)
                loss = upper_loss_gnn_CDR(
                    batch_val_y,
                    backbone_model,
                    spec_all,
                    params[0],
                    hparams[0],
                    N,
                    soap_loss,
                )
                f_y = torch.autograd.grad(loss, params, create_graph=False)[0]
                del loss; torch.cuda.empty_cache()

                batch_train_y1, train_iter = next_batch(self.conf.train_loader, train_iter)
                loss = lower_loss_gnn_CDR(
                    batch_train_y1,
                    backbone_model,
                    spec_all,
                    params[0],
                    hparams[0],
                    N,
                    soap_loss,
                )
                gy_y = torch.autograd.grad(loss, params, create_graph=False)[0]

                epoch_train_loss_sum += loss.item()
                del loss; torch.cuda.empty_cache()
                epoch_train_steps += 1

                if global_step == 0:
                    v_1.data = torch.clone(f_y.data).detach()
                    v_2.data = torch.clone(gy_y.data).detach()
                else:
                    v_1.data = (1 - self.alpha * self.eta) * (v_1.data) + (
                        self.alpha * self.eta
                    ) * f_y.data
                    v_2.data = (1 - self.alpha * self.eta) * (v_2.data) + (
                        self.alpha * self.eta
                    ) * gy_y.data
                m_y = v_1.data + (1 / self.rho) * v_2.data

                # =====compute z=======
                batch_train_z, train_iter = next_batch(self.conf.train_loader, train_iter)
                loss = lower_loss_gnn_CDR(
                    batch_train_z,
                    backbone_model,
                    spec_all,
                    z,
                    hparams[0],
                    N,
                    soap_loss,
                )
                gz_z = torch.autograd.grad(loss, z, create_graph=False)[0]
                del loss; torch.cuda.empty_cache()

                if global_step == 0:
                    w_1.data = torch.clone(gz_z.data).detach()
                else:
                    w_1.data = (1 - self.alpha * self.eta) * (w_1.data) + (
                        self.alpha * self.eta
                    ) * gz_z.data
                m_z = (1 / self.rho) * w_1.data

                ############# Add muon ############
                # m_x = newtonschulz5(m_x.reshape(N, N)).reshape(-1)
                mxM = m_x.view(N, N)
                mxM = mxM - torch.diag_embed(torch.diagonal(mxM))
                m_x = newtonschulz5(mxM).reshape(-1)

                m_y = apply_muon_to_vec_update(m_y, sl_all, backbone_model)
                m_z = apply_muon_to_vec_update(m_z, sl_all, backbone_model)

                # hparams[0].data = hparams[0].data.reshape(N, N)
                # hparams[0].data = ((hparams[0].data + hparams[0].data.T) / 2).reshape(-1)

                ############# Update x y z ############
                hparams[0].data -= self.eta * m_x.data * self.eta_x_expand
                params[0].data -= self.eta * m_y.data
                z.data -= self.eta * m_z.data

                with torch.no_grad():
                    H = hparams[0].data.view(N, N)
                    H = 0.5 * (H + H.T)
                    H.fill_diagonal_(0.0)
                    hparams[0].data.copy_(H.view(-1))

                # ========= end of iter ================
                step_time = time.time() - step_start_time
                total_time += step_time
                global_step += 1

            # ============ evaluation =======
            hparams_mean = torch.clone(hparams[0]).detach()
            params_mean = torch.clone(params[0]).detach()

            # print(hparams_mean)
            # print(vec_to_symmetric_mask(torch.sigmoid(hparams_mean), N))
            # print((torch.sigmoid(hparams_mean)).reshape(N, N))
            # print()

            val_loss, val_acc, val_f1, val_auc, val_auprc, _, _, y_true_val, y_score_val = eval_acc_gnn(
                self.conf.val_loader,
                backbone_model,
                spec_all,
                params_mean,
                hparams_mean,
                N,
            )
            tau = _best_f1_threshold(y_true_val, y_score_val[:, 1]) if y_score_val.shape[1] == 2 else None
            (
                test_loss,
                test_acc,
                test_f1,
                test_auc,
                test_auprc,
                test_cm,
                test_f1_each_class,
                _,
                _,
            ) = eval_acc_gnn(
                self.conf.test_loader,
                backbone_model,
                spec_all,
                params_mean,
                hparams_mean,
                N,
                threshold=tau,
            )

            if val_auprc >= best_val_auprc:
                best_val_f1 = float(val_f1)
                best_val_auprc = float(val_auprc)
                best_epoch = int(epoch)

                best_test_acc_at_best_val = float(test_acc)
                best_test_f1_at_best_val = float(test_f1)  # ✅
                best_test_auc_at_best_val = float(test_auc)  # ✅
                best_test_auprc_at_best_val = float(test_auprc)

                best_test_cm = test_cm
                best_test_f1_each_class = test_f1_each_class

                torch.save(
                    {"backbone": params_mean, "mask": hparams_mean},
                    params_path,
                )

                print(
                    f"save params (Val AUPRC: {val_auprc:.4f}, F1: {val_f1:.4f}):",
                    params_path,
                )

            train_loss_epoch = epoch_train_loss_sum / max(epoch_train_steps, 1)

            self.result["train_loss"][epoch] = train_loss_epoch
            self.result["upper_loss"][epoch] = val_loss
            self.result["val_acc"][epoch] = val_acc
            self.result["val_f1"][epoch] = val_f1
            self.result["val_auc"][epoch] = val_auc
            self.result["val_auprc"][epoch] = val_auprc
            self.result["test_loss"][epoch] = test_loss
            self.result["test_acc"][epoch] = test_acc
            self.result["test_f1"][epoch] = test_f1
            self.result["test_auc"][epoch] = test_auc
            self.result["test_auprc"][epoch] = test_auprc
            self.result["total_time"][epoch] = total_time

            if epoch % self.eval_interval == 0 or epoch == num_epochs:
                print(
                    f"epoch={epoch}, time={total_time:.3f}\n"
                    f"Train Loss={train_loss_epoch:.4f}, Val Loss={val_loss:.4f} | "
                    f"Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}\n"
                    f"F1 Score: Val={val_f1:.4f}, Test={test_f1:.4f} | "
                    f"AUC: Val={val_auc:.4f}, Test={test_auc:.4f} | "
                    f"AUPRC: Val={val_auprc:.4f}, Test={test_auprc:.4f}"
                )
                # check_stats("hparams", hparams[0])
                # check_stats("params", params[0])
                # check_stats("z", z)
                # check_stats("m_x", m_x)
                # check_stats("m_y", m_y)
                # check_stats("m_z", m_z)
                print()

                sys.stdout.flush()

            torch.cuda.empty_cache()

        print("total time = {}".format(total_time))

        with open(file_addr, "wb") as f:
            pickle.dump(self.result, f)
        print("save file = ", file_addr)

        #### plot matrix
        plot_mask(save_folder, N)

        save_fig_cdr(
            num_epochs,
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
            "best_val_auprc": float(best_val_auprc),
            "test_f1_at_best_val": float(best_test_f1_at_best_val),
            "test_acc_at_best_val": float(best_test_acc_at_best_val),
            "test_auc_at_best_val": float(best_test_auc_at_best_val),
            "test_auprc_at_best_val": float(best_test_auprc_at_best_val),
            "final_test_f1": float(self.result["test_f1"][num_epochs]),
            "save_folder": save_folder,
            "params_path": params_path,
            "CM": best_test_cm.tolist(),
        }


def check_stats(name, t):
    with torch.no_grad():
        print(
            f"{name}: "
            f"norm={t.norm().item():.4e}, "
            f"max|.|={t.abs().max().item():.4e}, "
            # f"has_nan={torch.isnan(t).any().item()}, "
            # f"has_inf={torch.isinf(t).any().item()}"
        )


def apply_muon_to_vec_update(vec_update, sl_all, model, steps=5, eps=1e-7):
    """
    vec_update: 1D tensor (比如 m_y 或 m_z)
    sl_all: dict[name] = slice
    model: backbone_model，用来拿每个参数的真实 shape
    """
    named_params = dict(model.named_parameters())
    out = vec_update.clone()
    for name, sl in sl_all.items():
        if name not in named_params:
            continue
        p = named_params[name]
        if any(k in name for k in ("classifier", "head", "embed")):
            continue
        if p.ndim == 2:
            G = out[sl].view_as(p)
            # muon / orthogonalize
            G2 = newtonschulz5(G)
            out[sl] = G2.reshape(-1)
    return out


def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
