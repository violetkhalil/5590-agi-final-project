# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join
import argparse
import torch
import numpy as np


def get_args():
    ROOT_DIRECTORY = "./"

    # feed them to the parser.
    parser = argparse.ArgumentParser(description="PyTorch Training for EHR")

    # add arguments.
    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--remote_exec", default=False, type=str2bool)
    parser.add_argument("--task", default=None, type=str)
    parser.add_argument(
        "--cdr_pair",
        default="default",
        type=str,
        help="binary CDR pair: ADNI {NC_SMC, SMC_MCI}, OASIS {NC_MCI, MCI_AD}, default keeps labels {0,1}",
    )

    ##########
    # -------------- GNN-related config (from CONFIG) -----------------
    parser.add_argument("--manual_seed", type=int, default=42, help="manual seed")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--n_nodes", type=int, default=82)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--msg_mlp_dim", type=int, default=64)
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of message-passing layers"
    )
    parser.add_argument("--readout_dim", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)

    # data split
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--rho_coeff", type=float, default=0.2)

    # dataset.
    parser.add_argument("--data", default="cifar10", type=str, help="dataset")
    parser.add_argument(
        "--scalar", default="standard_1", type=str, help="dataset normalization"
    )
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--imratio", default=0.1, type=float, help="imbalance ratio")
    parser.add_argument("--pin_memory", default=True, type=str2bool)

    parser.add_argument("--k_fold", default=5, type=int, help="k fold cross validation")

    # model
    parser.add_argument("--arch", default="resnet20", help="model architecture")
    parser.add_argument(
        "--hidden_feature", default=100, type=int, help="size of hidden feature"
    )
    parser.add_argument("--n_MLP_layers", type=int, default=1)
    parser.add_argument("--n_GNN_layers", type=int, default=2)
    parser.add_argument(
        "--pooling", type=str, choices=["sum", "concat", "mean"], default="sum"
    )
    parser.add_argument("--num_epochs", default=100, type=int, help="number of epochs")
    parser.add_argument(
        "--inner_steps", default=1, type=int, help="number of inner steps"
    )
    parser.add_argument("--y_loop", default=10, type=int, help="number of y lopp")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--rho", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)

    parser.add_argument("--eta_x_expand", default=10, type=float)

    parser.add_argument(
        "--beta_primal",
        default=0.1,
        type=float,
        help="beta: learning rate for primal variables",
    )
    parser.add_argument(
        "--beta_sq_primal",
        default=0.1,
        type=float,
        help="beta_sq: learning rate for primal variables",
    )
    parser.add_argument(
        "--beta_dual",
        default=0.1,
        type=float,
        help="beta: learning rate for dual variables",
    )
    parser.add_argument(
        "--gamma_primal",
        default=0.1,
        type=float,
        help="gamma: learning rate for primal variables",
    )
    parser.add_argument(
        "--gamma_dual",
        default=0.1,
        type=float,
        help="gamma: learning rate for dual variables",
    )
    parser.add_argument(
        "--alpha_primal",
        default=0.99,
        type=float,
        help="alpha coefficient for primal variables",
    )
    parser.add_argument(
        "--reg_coeff",
        default=0.001,
        type=float,
        help="coefficient for nonconvex regularizer",
    )

    parser.add_argument("--margin", default=1.0, type=float, help="margin")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")

    # baseline-only optimizer hparams (used by Trainer_ALTER_CDR for ALTER / BrainNETTF)
    parser.add_argument("--baseline_lr", default=1e-4, type=float,
                        help="lr for ALTER/BrainNETTF Adam optimizer")
    parser.add_argument("--baseline_weight_decay", default=1e-4, type=float,
                        help="weight_decay for ALTER/BrainNETTF Adam optimizer")

    parser.add_argument("--create_graph", type=str2bool, default=True)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="dscgda_gp")
    parser.add_argument("--eta", default=0.1, type=float, help="eta")
    parser.add_argument("--beta", default=0.1, type=float, help="beta")
    parser.add_argument("--epsilon", default=0.1, type=float, help="epsilon")
    parser.add_argument(
        "--alpha_eta_product", default=0.9, type=float, help="storm hyperparam"
    )

    # the topology of the decentralized network.
    parser.add_argument("--graph_topology", default="complete", type=str)

    """meta info."""
    parser.add_argument("--user", type=str, default="gao")
    parser.add_argument(
        "--project", type=str, default="distributed_adam_type_algorithm"
    )
    parser.add_argument("--experiment", type=str, default=None)

    # device
    parser.add_argument("--use_ipc", type=str2bool, default=False)
    parser.add_argument("--hostfile", type=str, default="iccluster/hostfile")
    parser.add_argument("--mpi_path", type=str, default="$HOME/.openmpi")
    parser.add_argument("--mpi_env", type=str, default=None)
    parser.add_argument(
        "--python_path",
        type=str,
        default="/home/user/conda/envs/pytorch-py3.6/bin/python",
    )

    parser.add_argument(
        "--n_mpi_process", default=1, type=int, help="# of the main process."
    )
    parser.add_argument(
        "--n_sub_process",
        default=1,
        type=int,
        help="# of subprocess for each mpi process.",
    )
    parser.add_argument("--world", default=None, type=str)
    parser.add_argument("--on_cuda", type=str2bool, default=True)
    parser.add_argument("--comm_device", type=str, default="cuda")
    parser.add_argument("--local_rank", default=None, type=str)
    parser.add_argument("--clean_python", default=False, type=str2bool)

    # new for compressor
    parser.add_argument("--comm_op", type=str, default=None)

    parser.add_argument("--primal_rho", default=0.1, type=float, help="lr")
    parser.add_argument("--dual_rho", default=0.1, type=float, help="lr")
    parser.add_argument("--compress_ratio", default=0.1, type=float, help="lr")
    parser.add_argument("--quantize_level", type=int, default=16, help="manual seed")
    parser.add_argument("--is_biased", type=str2bool, default=True)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--consensus_stepsize", default=0.1, type=float, help="lr")
    parser.add_argument("--lr_gamma", default=0.33, type=float, help="lr")
    parser.add_argument("--heter", default=0, type=float, help="whether use heter data")
    # parse conf.
    conf = parser.parse_args()

    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
