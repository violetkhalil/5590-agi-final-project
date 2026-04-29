import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
import time
import torch.nn.functional as F
import sys
import os
import torch.distributed as dist
import pickle
from decimal import Decimal
import torch.nn as nn
from Model.gnn import *


class Trainer_DecBiFirstOrder_pl_pruning_gnn:
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
            "total_n_bits",
            "total_time",
        ]
        for k in k_list:
            self.result[k] = np.zeros(self.conf.num_epochs + 1)

    def train(self):
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

        torch.set_default_dtype(torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_val_acc = -1.0
        best_epoch = -1
        best_test_acc_at_best_val = -1.0
        N = self.conf.N

        num_epochs = self.conf.num_epochs
        pre_epochs = num_epochs // 2

        folder_name = "epsilon-{}_eta-{}_rho-{}_momentum-{}_hidden_dim-{}_dropout-{}_batchsize-{}_eta_x_expand-{}".format(
            self.conf.epsilon,
            self.eta,
            self.rho,
            self.conf.alpha_eta_product,
            self.conf.hidden_dim,
            self.conf.dropout,
            self.conf.batch_size,
            self.conf.eta_x_expand,
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
            weight_decay=0.00001,
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

            val_loss, val_acc = evaluate(self.conf.val_loader)
            test_loss, test_acc = evaluate(self.conf.test_loader)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"pretrained_backbone_state_dict": backbone_model.state_dict()}, pretrain_path
                )
                print("save params:", pretrain_path)

            total_time += time.time() - epoch_start

            self.result["train_loss"][epoch] = train_loss
            self.result["train_acc"][epoch] = train_acc
            self.result["upper_loss"][epoch] = val_loss
            # self.result["val_loss"][epoch] = val_loss
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

        hparams_vec0 = torch.ones(N * N, device=device)
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
        train_loss, train_acc, *_ = eval_acc_gnn(
            self.conf.train_loader,
            backbone_model,
            spec_all,
            params[0].detach().clone(),
            hparams[0].detach().clone(),
            N,
        )
        val_loss, val_acc, *_ = eval_acc_gnn(
            self.conf.val_loader,
            backbone_model,
            spec_all,
            params[0].detach().clone(),
            hparams[0].detach().clone(),
            N,
        )
        test_loss, test_acc, *_ = eval_acc_gnn(
            self.conf.test_loader,
            backbone_model,
            spec_all,
            params[0].detach().clone(),
            hparams[0].detach().clone(),
            N,
        )

        self.result["train_loss"][0] = train_loss
        self.result["upper_loss"][0] = val_loss
        self.result["val_acc"][0] = val_acc
        self.result["test_loss"][0] = test_loss
        self.result["test_acc"][0] = test_acc

        def next_batch(loader, it):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            return batch, it

        global_step = 0
        best_val_acc = 0

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
                loss = upper_loss_gnn(
                    batch_val,
                    backbone_model,
                    spec_all,
                    params[0],
                    hparams[0],
                    N,
                )
                fy_x = torch.autograd.grad(loss, hparams, create_graph=True)[0]

                batch_train1, train_iter = next_batch(self.conf.train_loader, train_iter)
                loss = lower_loss_gnn(
                    batch_train1,
                    backbone_model,
                    spec_all,
                    params[0],
                    hparams[0],
                    N,
                )
                gy_x = torch.autograd.grad(loss, hparams, create_graph=True)[0]

                batch_train2, train_iter = next_batch(self.conf.train_loader, train_iter)
                loss = lower_loss_gnn(
                    batch_train2,
                    backbone_model,
                    spec_all,
                    z,
                    hparams[0],
                    N,
                )
                gz_x = torch.autograd.grad(loss, hparams, create_graph=True)[0]

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
                loss = upper_loss_gnn(
                    batch_val_y,
                    backbone_model,
                    spec_all,
                    params[0],
                    hparams[0],
                    N,
                )
                f_y = torch.autograd.grad(loss, params, create_graph=True)[0]

                batch_train_y1, train_iter = next_batch(self.conf.train_loader, train_iter)
                loss = lower_loss_gnn(
                    batch_train_y1,
                    backbone_model,
                    spec_all,
                    params[0],
                    hparams[0],
                    N,
                )
                gy_y = torch.autograd.grad(loss, params, create_graph=True)[0]

                epoch_train_loss_sum += loss.item()
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
                loss = lower_loss_gnn(
                    batch_train_z,
                    backbone_model,
                    spec_all,
                    z,
                    hparams[0],
                    N,
                )
                gz_z = torch.autograd.grad(loss, z, create_graph=True)[0]

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

            val_loss, val_acc, *_ = eval_acc_gnn(
                self.conf.val_loader,
                backbone_model,
                spec_all,
                params_mean,
                hparams_mean,
                N,
            )
            test_loss, test_acc, *_ = eval_acc_gnn(
                self.conf.test_loader,
                backbone_model,
                spec_all,
                params_mean,
                hparams_mean,
                N,
            )

            if val_acc >= best_val_acc:
                best_val_acc = float(val_acc)
                best_epoch = int(epoch)
                best_test_acc_at_best_val = float(test_acc)
                torch.save(
                    {
                        "backbone": params_mean,
                        "mask": hparams_mean,
                    },
                    params_path,
                )
                print("save params:", params_path)

            train_loss_epoch = epoch_train_loss_sum / max(epoch_train_steps, 1)

            self.result["train_loss"][epoch] = train_loss_epoch
            self.result["upper_loss"][epoch] = val_loss
            self.result["val_acc"][epoch] = val_acc
            self.result["test_loss"][epoch] = test_loss
            self.result["test_acc"][epoch] = test_acc
            self.result["total_time"][epoch] = total_time

            if epoch % self.eval_interval == 0 or epoch == num_epochs:
                print(
                    f"epoch={epoch} time={total_time:.3f} "
                    f"Train Loss={train_loss_epoch:.4f}, "
                    f"Upper Loss={val_loss:.4f}, Val Acc={val_acc:.4f} "
                    f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
                )
                # check_stats("hparams", hparams[0])
                # check_stats("params", params[0])
                # check_stats("z", z)
                # check_stats("m_x", m_x)
                # check_stats("m_y", m_y)
                # check_stats("m_z", m_z)
                print()

                sys.stdout.flush()

        print("total time = {}".format(total_time))

        with open(file_addr, "wb") as f:
            pickle.dump(self.result, f)
        print("save file = ", file_addr)

        save_fig_bilevel(num_epochs, self.result, save_folder, best_test_acc_at_best_val)
        #### plot matrix
        plot_mask(save_folder, N)

        return {
            "fold": getattr(self.conf, "fold", None),
            "best_epoch": best_epoch,
            "best_val_acc": float(best_val_acc),
            "test_acc_at_best_val": float(best_test_acc_at_best_val),
            "final_test_acc": float(self.result["test_acc"][num_epochs]),
            "save_folder": save_folder,
            "params_path": params_path,
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
