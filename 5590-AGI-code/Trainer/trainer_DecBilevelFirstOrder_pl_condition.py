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

from Utils.utils import consistent_model, get_model_stat
from Utils.zyh_logging import (
    display_training_stat,
    display_eval_stat,
    display_test_stat,
)
from Model.auc_loss import CrossEntropyBinaryLoss, AUCMLoss
from Optimizer.dsgda_gp import DSGDAGP
from Optimizer.zyh_c_dsgdam import D_C_SGDAM
from Optimizer.zyh_sgdam_no_errorFeedBack import SGDAM_NEF
import Utils.communication as comm

from Utils.zyh_hyperparameter_pl_condition import generate_imbalance
from Utils.zyh_hyperparameter_pl_condition import get_n_bits
from Utils.zyh_hyperparameter_pl_condition import get_data_matrix
from Utils.zyh_hyperparameter_pl_condition import tonp
from Utils.zyh_hyperparameter_pl_condition import eval_acc
from Utils.zyh_hyperparameter_pl_condition import get_full_data
from Utils.zyh_hyperparameter_pl_condition import lower_loss
from Utils.zyh_hyperparameter_pl_condition import upper_loss


class Trainer_DecBiFirstOrder_pl:
    def __init__(self, dataset, conf):
        print("it is Trainer: Decentralized Bilevel First Order")
        self.dataset = dataset
        self.conf = conf
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()

        self.eval_interval = 1
        # if self.conf.data == 'covtype':
        #     self.eta = 0.0004
        # else:
        #     self.eta = 0.0004
        # self.eta = round(self.conf.epsilon ** 3, 8)
        self.eta = float(Decimal(str(self.conf.epsilon)) ** 3)
        # self.eta = self.conf.lr
        self.rho = float(Decimal(str(self.conf.epsilon)) * Decimal("0.2"))
        # self.rho=self.conf.lr
        self.alpha = self.conf.alpha_eta_product / (self.eta**2)

        # self.im_list = self.conf.im_list

        self.result = {}
        k_list = [
            "local_train_loss",
            "full_train_loss",
            "local_val_loss",
            "full_val_loss",
            "test_loss",
            "test_acc",
            "total_n_bits",
            "total_time",
        ]
        for k in k_list:
            self.result[k] = np.zeros(self.conf.num_epochs + 1)

        if self.conf.graph_topology == "random" or self.conf.graph_topology == "torus":
            self.ad_matrix = self.conf.graph._mixing_matrix
            self.edge_num = np.count_nonzero(self.ad_matrix) - self.ad_matrix.shape[0]
            print("edge num: ", self.edge_num)
        if self.conf.graph_topology == "ring":
            self.edge_num = self.conf.graph.n_edges
            print("edge num: ", self.edge_num)
        if self.conf.graph_topology == "complete":
            self.edge_num = self.conf.graph.n_edges
            print("edge num: ", self.edge_num)
        self.world_aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=dict(
                (rank, 1.0 / conf.graph.n_nodes) for rank in conf.graph.ranks
            ),
            aggregator_type="centralized",
        )

        self.aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=self.neighbors_info,
            aggregator_type="decentralized",
        )

    def local_logistic_loss(self, x_train_batch, y_train_batch):
        def f(w):
            return F.binary_cross_entropy_with_logits(x_train_batch @ w, y_train_batch)

        return f

    def train(self):
        default_tensor_str = "torch.FloatTensor"
        torch.set_default_tensor_type(default_tensor_str)

        seed = self.conf.manual_seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ============ data =======
        x_test, y_test = get_full_data(self.dataset.testloader)
        x_train, y_train = get_full_data(self.dataset.trainloader2)
        x_val, y_val = get_full_data(self.dataset.valloader2)

        self.conf.training_size = y_train.size()[0]
        self.conf.validation_size = y_val.size()[0]

        # ============ params=======
        feature_dimension = x_test.shape[1]
        hidden_dimension = self.conf.hidden_feature
        # num_class = torch.unique(y_test).size(0)
        num_class = 1
        if self.conf.data == "mnist":
            num_class = 10

        param_x1 = torch.empty((feature_dimension), requires_grad=True)
        nn.init.uniform_(param_x1)
        param_x2 = torch.empty((hidden_dimension), requires_grad=True)
        nn.init.uniform_(param_x2)
        x_combined = torch.cat([param_x1, param_x2], dim=0)

        param_y1 = torch.empty(
            (feature_dimension, hidden_dimension), requires_grad=True
        )
        nn.init.xavier_uniform_(param_y1)
        param_y2 = torch.empty((hidden_dimension, num_class), requires_grad=True)
        nn.init.xavier_uniform_(param_y2)
        y1_flattened = param_y1.flatten()
        y2_flattened = param_y2.flatten()
        y_combined = torch.cat([y1_flattened, y2_flattened], dim=0)

        hparams = [torch.clone(x_combined).detach().requires_grad_(True)]  # lambda, x
        params = [torch.clone(y_combined).detach().requires_grad_(True)]  # tau, y
        z = torch.clone(y_combined).detach().requires_grad_(True)

        hparams_prior = [
            torch.clone(x_combined).detach().requires_grad_(True)
        ]  # lambda, x
        params_prior = [torch.clone(y_combined).detach().requires_grad_(True)]  # tau, y
        z_prior = torch.clone(y_combined).detach().requires_grad_(True)

        # x_dimension = feature_dimension + hidden_dimension
        # y_dimension = feature_dimension * hidden_dimension + hidden_dimension * num_class
        #
        # hparams = [torch.randn(x_dimension).requires_grad_(True)]  # lambda, x
        # params = [torch.randn(y_dimension).requires_grad_(True)]  # tau, y
        # z = torch.randn(y_dimension).requires_grad_(True)
        #
        # hparams_prior = [torch.randn(x_dimension).requires_grad_(True)]  # lambda, x
        # params_prior = [torch.randn(y_dimension).requires_grad_(True)]  # tau, y
        # z_prior = torch.randn(y_dimension).requires_grad_(True)

        u_1 = torch.zeros_like(hparams[0])
        u_2 = torch.zeros_like(hparams[0])
        u_3 = torch.zeros_like(hparams[0])
        v_1 = torch.zeros_like(params[0])
        v_2 = torch.zeros_like(params[0])
        w_1 = torch.zeros_like(z)

        # ============ evaluation =======
        hparams_mean = torch.clone(hparams[0]).detach()
        tmp = self.world_aggregator._agg(hparams_mean, op="avg")
        dist.barrier()
        hparams_mean = torch.clone(tmp).detach()

        params_mean = torch.clone(params[0]).detach()
        tmp = self.world_aggregator._agg(params_mean, op="avg")
        dist.barrier()
        params_mean = torch.clone(tmp).detach()

        test_loss = upper_loss(
            x_test, y_test, params_mean, feature_dimension, hidden_dimension, num_class
        )
        if self.conf.data == "mnist":
            test_acc = eval_acc(
                x_test,
                y_test,
                params_mean,
                feature_dimension,
                hidden_dimension,
                num_class,
            )
        else:
            test_acc = eval_acc(
                x_test,
                y_test,
                params_mean,
                feature_dimension,
                hidden_dimension,
                num_class,
            )
        self.result["test_loss"][0] = tonp(test_loss)
        self.result["test_acc"][0] = test_acc

        total_n_bits = 0
        total_time = 0
        outer_steps = self.conf.num_epochs

        for o_step in range(outer_steps):
            step_start_time = time.time()

            # ============update x ==============
            batch_index = np.random.permutation(np.arange(self.conf.training_size))[
                0 : self.conf.batch_size
            ]
            x_train_batch, y_train_batch = x_train[batch_index], y_train[batch_index]
            loss = lower_loss(
                x_train_batch,
                y_train_batch,
                hparams[0],
                params[0],
                feature_dimension,
                hidden_dimension,
                num_class,
            )
            gy_x = torch.autograd.grad(loss, hparams, create_graph=True)[0]

            batch_index = np.random.permutation(np.arange(self.conf.training_size))[
                0 : self.conf.batch_size
            ]
            x_train_batch_2, y_train_batch_2 = (
                x_train[batch_index],
                y_train[batch_index],
            )
            loss = lower_loss(
                x_train_batch_2,
                y_train_batch_2,
                hparams[0],
                z,
                feature_dimension,
                hidden_dimension,
                num_class,
            )
            gz_x = torch.autograd.grad(loss, hparams, create_graph=True)[0]

            u_1 = torch.zeros_like(hparams[0])

            if o_step == 0:
                u_2.data = torch.clone(gy_x).detach()
                u_3.data = torch.clone(gz_x).detach()
                m_x = u_1.data + (1 / self.rho) * (u_2.data - u_3.data)
                U_x = torch.clone(m_x.data).detach()
            else:
                loss = lower_loss(
                    x_train_batch,
                    y_train_batch,
                    hparams_prior[0],
                    params_prior[0],
                    feature_dimension,
                    hidden_dimension,
                    num_class,
                )
                gy_x_prior = torch.autograd.grad(
                    loss, hparams_prior, create_graph=True
                )[0]

                loss = lower_loss(
                    x_train_batch_2,
                    y_train_batch_2,
                    hparams_prior[0],
                    z_prior,
                    feature_dimension,
                    hidden_dimension,
                    num_class,
                )
                gz_x_prior = torch.autograd.grad(
                    loss, hparams_prior, create_graph=True
                )[0]

                u_2.data = (1 - self.alpha * self.eta**2) * (
                    u_2.data - gy_x_prior.data
                ) + gy_x.data
                u_3.data = (1 - self.alpha * self.eta**2) * (
                    u_3.data - gz_x_prior.data
                ) + gz_x.data

                m_x_prior = torch.clone(m_x).detach()
                m_x = u_1.data + (1 / self.rho) * (u_2.data - u_3.data)

                tmp = torch.clone(U_x.data).detach()
                tmp1 = self.aggregator._agg(tmp, op="weighted")
                total_n_bits += get_n_bits(U_x) * self.edge_num
                dist.barrier()

                U_x.data = torch.clone(tmp1).detach()
                U_x.data += m_x.data - m_x_prior.data

            # =========update y=========
            batch_index = np.random.permutation(np.arange(self.conf.validation_size))[
                0 : self.conf.batch_size
            ]
            x_val_batch, y_val_batch = x_val[batch_index], y_val[batch_index]
            loss = upper_loss(
                x_val_batch,
                y_val_batch,
                params[0],
                feature_dimension,
                hidden_dimension,
                num_class,
            )
            f_y = torch.autograd.grad(loss, params, create_graph=True)[0]

            batch_index = np.random.permutation(np.arange(self.conf.training_size))[
                0 : self.conf.batch_size
            ]
            x_train_batch, y_train_batch = x_train[batch_index], y_train[batch_index]
            loss = lower_loss(
                x_train_batch,
                y_train_batch,
                hparams[0],
                params[0],
                feature_dimension,
                hidden_dimension,
                num_class,
            )
            gy_y = torch.autograd.grad(loss, params, create_graph=True)[0]

            if o_step == 0:
                v_1.data = torch.clone(f_y.data).detach()
                v_2.data = torch.clone(gy_y.data).detach()
                m_y = v_1.data + (1 / self.rho) * v_2.data
                U_y = torch.clone(m_y.data).detach()
            else:
                loss = upper_loss(
                    x_val_batch,
                    y_val_batch,
                    params_prior[0],
                    feature_dimension,
                    hidden_dimension,
                    num_class,
                )
                f_y_prior = torch.autograd.grad(loss, params_prior, create_graph=True)[
                    0
                ]

                loss = lower_loss(
                    x_train_batch,
                    y_train_batch,
                    hparams_prior[0],
                    params_prior[0],
                    feature_dimension,
                    hidden_dimension,
                    num_class,
                )
                gy_y_prior = torch.autograd.grad(loss, params_prior, create_graph=True)[
                    0
                ]

                v_1.data = (1 - self.alpha * self.eta**2) * (
                    v_1.data - f_y_prior.data
                ) + f_y.data
                v_2.data = (1 - self.alpha * self.eta**2) * (
                    v_2.data - gy_y_prior.data
                ) + gy_y.data
                m_y_prior = torch.clone(m_y).detach()
                m_y = v_1.data + (1 / self.rho) * v_2.data

                tmp = torch.clone(U_y.data).detach()
                tmp1 = self.aggregator._agg(tmp, op="weighted")
                total_n_bits += get_n_bits(U_y) * self.edge_num
                dist.barrier()

                U_y.data = torch.clone(tmp1).detach()
                U_y.data += m_y.data - m_y_prior.data

            # communication and update

            # =====compute z=======
            batch_index = np.random.permutation(np.arange(self.conf.training_size))[
                0 : self.conf.batch_size
            ]
            x_train_batch_2, y_train_batch_2 = (
                x_train[batch_index],
                y_train[batch_index],
            )
            loss = lower_loss(
                x_train_batch_2,
                y_train_batch_2,
                hparams[0],
                z,
                feature_dimension,
                hidden_dimension,
                num_class,
            )
            gz_z = torch.autograd.grad(loss, z, create_graph=True)[0]

            if o_step == 0:
                w_1.data = torch.clone(gz_z.data).detach()
                m_z = (1 / self.rho) * w_1.data
                U_z = torch.clone(m_z.data).detach()
            else:
                loss = lower_loss(
                    x_train_batch_2,
                    y_train_batch_2,
                    hparams_prior[0],
                    z_prior,
                    feature_dimension,
                    hidden_dimension,
                    num_class,
                )
                gz_z_prior = torch.autograd.grad(loss, z_prior, create_graph=True)[0]

                w_1.data = (1 - self.alpha * self.eta**2) * (
                    w_1.data - gz_z_prior.data
                ) + gz_z.data

                m_z_prior = torch.clone(m_z).detach()
                m_z = (1 / self.rho) * w_1.data

                tmp = torch.clone(U_z.data).detach()
                tmp1 = self.aggregator._agg(tmp, op="weighted")
                total_n_bits += get_n_bits(U_z) * self.edge_num
                dist.barrier()

                U_z.data = torch.clone(tmp1).detach()
                U_z.data += m_z.data - m_z_prior.data

            tmp = torch.clone(hparams[0]).detach()
            tmp1 = self.aggregator._agg(tmp, op="weighted")
            dist.barrier()
            total_n_bits += get_n_bits(hparams[0]) * self.edge_num

            hparams_prior[0].data = torch.clone(hparams[0].data).detach()
            hparams[0].data = torch.clone(tmp1).detach()
            hparams[0].data -= self.eta * U_x.data

            tmp = torch.clone(params[0]).detach()
            tmp1 = self.aggregator._agg(tmp, op="weighted")
            dist.barrier()
            total_n_bits += get_n_bits(params[0]) * self.edge_num

            params_prior[0].data = torch.clone(params[0].data).detach()
            params[0].data = torch.clone(tmp1).detach()
            params[0].data -= self.eta * U_y.data

            tmp = torch.clone(z).detach()
            tmp1 = self.aggregator._agg(tmp, op="weighted")
            dist.barrier()
            total_n_bits += get_n_bits(z) * self.edge_num

            z_prior.data = torch.clone(z.data).detach()
            z.data = torch.clone(tmp1).detach()
            z.data -= self.eta * U_z.data

            # ========= end of iter ================
            step_time = time.time() - step_start_time
            total_time += step_time

            # ============ evaluation =======
            hparams_mean = torch.clone(hparams[0]).detach()
            tmp = self.world_aggregator._agg(hparams_mean, op="avg")
            dist.barrier()
            hparams_mean = torch.clone(tmp).detach()

            params_mean = torch.clone(params[0]).detach()
            tmp = self.world_aggregator._agg(params_mean, op="avg")
            dist.barrier()
            params_mean = torch.clone(tmp).detach()

            test_loss = upper_loss(
                x_test,
                y_test,
                params_mean,
                feature_dimension,
                hidden_dimension,
                num_class,
            )
            if self.conf.data == "mnist":
                test_acc = eval_acc(
                    x_test,
                    y_test,
                    params_mean,
                    feature_dimension,
                    hidden_dimension,
                    num_class,
                )
            else:
                test_acc = eval_acc(
                    x_test,
                    y_test,
                    params_mean,
                    feature_dimension,
                    hidden_dimension,
                    num_class,
                )
            self.result["test_loss"][o_step + 1] = tonp(test_loss)
            self.result["test_acc"][o_step + 1] = test_acc
            self.result["total_n_bits"][o_step + 1] = total_n_bits
            self.result["total_time"][o_step + 1] = total_time

            if (
                (o_step + 1) % self.eval_interval == 0 or (o_step + 1) == outer_steps
            ) and self.rank == 0:
                print(
                    "o_step={} time={} Full Training Loss={} Full Test Loss={} Accuracy={}".format(
                        o_step + 1,
                        total_time,
                        self.result["full_train_loss"][o_step + 1],
                        self.result["test_loss"][o_step + 1],
                        self.result["test_acc"][o_step + 1],
                    )
                )
                sys.stdout.flush()

        if self.conf.graph.rank == 0:
            print("total time = {}".format(total_time))

            folder_name = "topology-{}_DecBiFirstOrder_eta-{}_rho-{}_momentum-{}_hidden-{}".format(
                self.conf.graph_topology,
                self.eta,
                self.rho,
                self.conf.alpha_eta_product,
                self.conf.hidden_feature,
            )

            # save_folder = os.path.join('./checkpoints_new/svm/{}'.format(self.conf.data), folder_name)
            if (
                self.conf.scalar == "norm"
                or self.conf.scalar == "minmax"
                or self.conf.scalar == "standard"
                or self.conf.scalar == "standard_1"
            ):
                save_folder = os.path.join(
                    "./checkpoints_pl_condition/svm/{}/{}".format(
                        self.conf.data, self.conf.scalar
                    ),
                    folder_name,
                )
            else:
                raise NotImplementedError

            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)

            file_addr = os.path.join(save_folder, "result.pkl")
            with open(file_addr, "wb") as f:
                pickle.dump(self.result, f)
