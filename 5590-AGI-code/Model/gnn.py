from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from torch_geometric.nn import GraphNorm
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import Tensor
from torch_sparse import (
    set_diag,
    SparseTensor,
    matmul,
    fill_diag,
    sum as sparsesum,
    mul,
)
import torch.nn as nn
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch.func import functional_call
from collections import OrderedDict

# from message_passing.message_passing import (
#     ModifiedMessagePassing as MessagePassing,
# )
from torch_geometric.nn import MessagePassing
from utils.utils import maybe_num_nodes, _remove_self_loops, _add_self_loops


# triu_idx = torch.triu_indices(N, N, offset=1)


def gcn_norm(
    edge_index,
    edge_flag,
    edge_attr=None,
    num_nodes=None,
    improved=False,
    do_add_self_loops=True,
    dtype=None,
):
    fill_value = 2.0 if improved else 1.0

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0, dtype=dtype)
        if do_add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        if do_add_self_loops:
            if isinstance(edge_index, Tensor):
                if isinstance(edge_flag, Tensor):
                    edge_index, edge_attr, edge_flag = _remove_self_loops(
                        edge_index, edge_attr, edge_flag
                    )
                    edge_index, edge_attr, edge_flag = _add_self_loops(
                        edge_index, edge_attr, edge_flag, num_nodes=num_nodes
                    )
                else:
                    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                    edge_index, edge_attr = add_self_loops(
                        edge_index, edge_attr, num_nodes=num_nodes
                    )
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_attr, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        return edge_index, deg_inv_sqrt[row] * edge_attr * deg_inv_sqrt[col], edge_flag


class MPConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        # add_self_loops: bool = False,
        normalize: bool = True,
        bias: bool = True,
    ):
        super(MPConv, self).__init__(aggr="add")

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.__explain__ = False

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.lin = torch.nn.Linear(out_channels * 2 + 1, out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_attr, edge_flag, edge_gate=None):
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight, edge_flag = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_flag,
                        edge_attr,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight, edge_flag)
                else:
                    edge_index, edge_weight, edge_flag = cache[0], cache[1], cache[2]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_attr,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # if edge_gate is not None:
        #     edge_weight = edge_weight * edge_gate

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # out = self.propagate(edge_index, edge_flag, x=x, edge_attr=edge_weight)
        out = self.propagate(edge_index, x=x, edge_attr=edge_weight)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr.view(-1, 1)], dim=1)
        return self.lin(msg)


class IBGConv(torch.nn.Module):
    def __init__(self, input_dim, args, num_classes):
        super(IBGConv, self).__init__()
        self.activation = torch.nn.ReLU()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        hidden_dim = args.hidden_dim
        num_layers = args.n_GNN_layers
        self.pooling = args.pooling
        self.dropout = args.dropout

        for i in range(num_layers):
            in_c = input_dim if i == 0 else hidden_dim
            out_c = hidden_dim
            self.convs.append(MPConv(in_c, out_c))
            self.norms.append(GraphNorm(out_c))

        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr, edge_flag, batch, edge_gate=None):
        z = x
        edge_attr = edge_attr.abs()
        for i, conv in enumerate(self.convs):
            z = conv(z, edge_index, edge_attr, edge_flag, edge_gate=edge_gate)
            z = self.norms[i](z, batch)
            if i != len(self.convs) - 1:
                z = F.relu(z)
                z = F.dropout(z, p=self.dropout, training=self.training)

        if self.pooling == "sum":
            g = global_add_pool(z, batch)
        elif self.pooling == "mean":
            g = global_mean_pool(z, batch)
        else:
            raise NotImplementedError("Pooling method not implemented")

        return self.classifier(g)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, n_classes=0):
        super(MLP, self).__init__()
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(activation())
        for _ in range(num_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(activation())
        self.net = torch.nn.Sequential(*self.net)
        self.shortcut = torch.nn.Linear(input_dim, hidden_dim)

        if n_classes != 0:
            self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out = self.net(x) + self.shortcut(x)
        if hasattr(self, "classifier"):
            return out, self.classifier(out)
        return out


class IBGNN(torch.nn.Module):
    def __init__(self, gnn, mlp, discriminator=lambda x, y: x @ y.t(), pooling="concat"):
        super(IBGNN, self).__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.pooling = pooling
        self.discriminator = discriminator

    def forward(self, data, mask_mat: torch.Tensor = None):

        x, edge_index, edge_attr, batch, edge_flag = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.edge_flag,
        )

        edge_gate = None
        edge_attr_used = edge_attr

        # if mask_mat is not None:
        #     col_l = col - data.ptr[g]


        g = self.gnn(x, edge_index, edge_attr, edge_flag, batch)
        # g = self.gnn(x, edge_index, edge_attr_used, edge_flag, batch, edge_gate=None)

        # if self.pooling == "concat":
        #     _, g = self.mlp(g)
        #     # log_logits = F.log_softmax(g, dim=-1)
        #     # return log_logits
        #     return g
        return g


# class MLP(nn.Module):
#     def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, out_dim),
#         )

#     def forward(self, x):
#         return self.net(x)


# class BrainNN(nn.Module):
#     """Edge-weight-aware GNN backbone.
#     - Node encoder over features
#     - L rounds of message passing with explicit w_ij in messages
#     - Sum readout + residual MLP
#     """

#     def __init__(
#         self,
#         d_node_in: int,
#         hidden: int,
#         msg_dim: int,
#         readout_dim: int,
#         num_layers: int,
#         num_classes: int,
#         dropout: float,
#     ):
#         super().__init__()
#         self.n_layers = num_layers
#         self.hidden = hidden
#         self.dropout = dropout

#         self.enc = nn.Sequential(
#             nn.Linear(d_node_in, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden),
#         )
#         # message MLP per layer
#         self.msg_mlps = nn.ModuleList(
#             [
#                 MLP(
#                     in_dim=hidden * 2 + 1,
#                     hidden=msg_dim,
#                     out_dim=hidden,
#                     dropout=dropout,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#         # readout
#         self.readout = MLP(
#             in_dim=hidden, hidden=readout_dim, out_dim=readout_dim, dropout=dropout
#         )
#         self.cls = nn.Linear(readout_dim, num_classes)

#     def message_pass(
#         self, H: torch.Tensor, W: torch.Tensor, layer_idx: int
#     ) -> torch.Tensor:
#         """H: [B, N, H], W: [B, N, N]"""
#         B, N, Hdim = H.shape
#         # Build all pair (i,j) tensors efficiently via broadcasting
#         Hi = H.unsqueeze(2).expand(B, N, N, Hdim)  # [B, N, N, H]
#         Hj = H.unsqueeze(1).expand(B, N, N, Hdim)  # [B, N, N, H]
#         Wij = W.unsqueeze(-1)  # [B, N, N, 1]
#         MsgIn = torch.cat([Hi, Hj, Wij], dim=-1)  # [B, N, N, 2H+1]
#         # compute messages per edge
#         m = self.msg_mlps[layer_idx](MsgIn)  # [B, N, N, H]
#         # aggregate over neighbors j (sum)
#         m = m.sum(dim=2)  # [B, N, H]
#         return F.relu(m)

#     def forward(self, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
#         """W: [B, N, N], X: [B, N, d]
#         returns logits: [B, C]
#         """
#         B, N, _ = W.shape
#         h = self.enc(X)  # [B, N, H]

#         for l in range(self.n_layers):
#             m = self.message_pass(h, W, l)
#             h = F.dropout(h + m, p=self.dropout, training=self.training)  # residual
#         # readout: sum over nodes
#         z_prime = h.sum(dim=1)  # [B, H]
#         z = self.readout(z_prime) + z_prime  # residual
#         logits = self.cls(z)
#         return logits


# class GlobalMask(nn.Module):
#     """Learn a global, shared mask M (logits) of shape [N, N].
#     Symmetrize and zero diag during forward.
#     """

#     def __init__(self, n_nodes: int):
#         super().__init__()
#         self.mask_logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))
#         nn.init.normal_(self.mask_logits, mean=0.0, std=0.02)

#     def forward(self) -> torch.Tensor:
#         M = torch.sigmoid(self.mask_logits)  # [N, N]
#         # symmetrize + zero diag
#         M = 0.5 * (M + M.T)
#         M = M - torch.diag(torch.diag(M))
#         return M

#     @staticmethod
#     def apply_mask(W: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
#         """Element-wise mask: W' = W ⊙ M. Shapes: W [B,N,N], M [N,N]."""
#         return W * M.unsqueeze(0)


def apply_mask(W: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    W: [B, N, N]
    M: [N, N]
    return Wm: [B, N, N]
    """
    return W * M.unsqueeze(0)


def vec_to_symmetric_mask(mask_vals: torch.Tensor, N) -> torch.Tensor:
    """
    mask_vals: 1D tensor, 已经是 [0,1] 的概率，长度 = N*(N-1)/2
    返回：对称的 [N, N] mask 矩阵（无对角）
    """
    device = mask_vals.device
    M = torch.zeros(N, N, device=device)
    triu_idx = torch.triu_indices(N, N, offset=1)
    idx = triu_idx.to(device)
    M[idx[0], idx[1]] = mask_vals
    M[idx[1], idx[0]] = mask_vals
    return M


# class GlobalMask(nn.Module):
#     def __init__(self, n_nodes):
#         super().__init__()
#         self.n = n_nodes
#         num_edges = n_nodes * (n_nodes - 1) // 2

#         self.mask_logits = nn.Parameter(torch.zeros(num_edges))
#         nn.init.normal_(self.mask_logits, 0, 0.02)

#         self.register_buffer("triu_idx", torch.triu_indices(n_nodes, n_nodes, offset=1))

#     def forward(self):
#         return self.vec_to_symmetric_mask(self.mask_logits)

#     def vec_to_symmetric_mask(self, hparams_vec_mask):
#         """
#         """
#         device = self.mask_logits.device
#         vals = torch.sigmoid(hparams_vec_mask.to(device))

#         M = torch.zeros(self.n, self.n, device=device)
#         M[self.triu_idx[0], self.triu_idx[1]] = vals
#         M = M + M.T
#         return M

#     @staticmethod
#     def apply_mask(W, M):
#         return W * M.unsqueeze(0)


def build_param_spec_all(model):
    return [(n, p.shape) for n, p in model.named_parameters()]


def pack_params_from_model(model, spec) -> torch.Tensor:
    flat = []
    pmap = dict(model.named_parameters())
    for name, _ in spec:
        flat.append(pmap[name].detach().reshape(-1))
    return torch.cat(flat, dim=0)


def flat_slices_from_spec(spec):
    pos = 0
    sl = {}
    for name, shape in spec:
        n = int(torch.tensor(shape).prod().item())
        sl[name] = slice(pos, pos + n)
        pos += n
    return sl, pos


# def evaluate(model: BrainNN, loader: DataLoader, device):
#     model.eval()
#     total_loss, total_acc, n = 0.0, 0.0, 0
#     with torch.no_grad():
#         for batch in loader:
#             W = batch.W.to(device)
#             X = batch.X.to(device)
#             y = batch.y.to(device)
#             logits = model(W, X)
#             loss = F.cross_entropy(logits, y)
#             b = y.size(0)
#             total_loss += loss.item() * b
#             total_acc += accuracy_from_logits(logits, y) * b
#             n += b
#     return total_loss / n, total_acc / n
# def evaluate(model: BrainNN, loader: DataLoader, device):
#     model.eval()
#     total_loss, total_acc, n = 0.0, 0.0, 0
#     with torch.no_grad():
#         for batch in loader:
#             W = batch.W.to(device)
#             X = batch.X.to(device)
#             y = batch.y.to(device)
#             logits = model(W, X)
#             loss = F.cross_entropy(logits, y)
#             b = y.size(0)
#             total_loss += loss.item() * b
#             total_acc += accuracy_from_logits(logits, y) * b
#             n += b
#     return total_loss / n, total_acc / n


# def masked_state_dict(model, spec_all, params_vec_full, gate_full, device):
#     out = OrderedDict()
#     pos = 0
#     for name, shape in spec_all:
#         n = int(torch.tensor(shape).prod().item())
#         w = (
#             (params_vec_full[pos : pos + n] * gate_full[pos : pos + n])
#             .view(shape)
#             .to(device)
#         )
#         out[name] = w
#         pos += n
#     return out


def forward_with_masked_params(model, params_state, W, X):
    return functional_call(model, params_state, (W, X), {})


def state_from_vec(model, spec_all, params_vec_full, device):
    """
    spec_all: [(name, shape), ...]
    params_vec_full: 1D tensor, 拼了所有参数
    返回: OrderedDict[name -> weight_tensor]
    """
    out = OrderedDict()
    pos = 0
    for name, shape in spec_all:
        n = int(torch.tensor(shape).prod().item())
        w = params_vec_full[pos : pos + n].view(shape).to(device)
        out[name] = w
        pos += n
    return out


def prune_edge_mask(edge_mask: Tensor, edge_flag: Tensor) -> Tensor:
    edge_mask = edge_mask.reshape(-1)
    catted_edge_mask = edge_mask.repeat(len(edge_flag))  # (B*6724,)

    edge_flag = np.concatenate(edge_flag, axis=0)  # (B*6724,)
    edge_flag = torch.from_numpy(edge_flag).to(edge_mask.device).bool()

    pruned_edge_mask = catted_edge_mask[edge_flag]  # (sum_E,)
    return pruned_edge_mask


def lower_loss_gnn(
    batch,
    backbone_model: IBGNN,
    spec_all,
    params_vec_full: torch.Tensor,
    hparams_vec_mask: torch.Tensor,
    N,
    l2_reg: float = 0.0,
) -> torch.Tensor:
    device = next(backbone_model.parameters()).device
    batch = batch.to(device)

    ############### refer original code  (W+W^T)/2
    # hparams_vec_mask = hparams_vec_mask.reshape(N, N)
    # hparams_vec_mask = (hparams_vec_mask + hparams_vec_mask.T) / 2
    H = hparams_vec_mask.view(N, N)
    H = 0.5 * (H + H.T)
    H = H - torch.diag_embed(torch.diagonal(H))

    pruned_edge_mask = prune_edge_mask(H, batch.edge_flag).to(device)
    explained_edge_attr = batch.edge_attr * pruned_edge_mask.view(1, -1).sigmoid()
    explained_edge_attr = explained_edge_attr.squeeze()
    batch.edge_attr = explained_edge_attr
    batch.edge_flag = pruned_edge_mask

    # # mask = hparams_vec_mask.to(device).view(N, N)
    # mask_prob = torch.sigmoid(hparams_vec_mask)  # [len_mask]
    # M = vec_to_symmetric_mask(mask_prob)  # [N, N]

    # print(M)

    params_state = state_from_vec(backbone_model, spec_all, params_vec_full, device)

    logits = functional_call(
        backbone_model,
        params_state,
        (batch,),
    )

    # 4. loss
    y = batch.y.long().view(-1)
    loss = F.cross_entropy(logits, y)

    if l2_reg > 0:
        loss = loss + l2_reg * params_vec_full.pow(2).mean()

    return loss


def upper_loss_gnn(
    batch,
    backbone_model: IBGNN,
    spec_all,
    params_vec_full: torch.Tensor,  # y
    hparams_vec_mask: torch.Tensor,  # m
    N,
) -> torch.Tensor:
    """
    用一个 Data batch 来计算 upper-level loss：
      L_upper(y, m) = CE(backbone(batch; y, m), batch.y)
    """
    device = next(backbone_model.parameters()).device
    batch = batch.to(device)

    params_state = state_from_vec(backbone_model, spec_all, params_vec_full, device)

    #####
    with torch.no_grad():
        logits_full = functional_call(
            backbone_model,
            params_state,
            (batch,),
        )
        pred_label = logits_full.argmax(dim=-1)  # [B]

    ############### refer original code
    # hparams_vec_mask = hparams_vec_mask.reshape(N, N)
    # hparams_vec_mask = (hparams_vec_mask + hparams_vec_mask.T) / 2
    H = hparams_vec_mask.view(N, N)
    H = 0.5 * (H + H.T)
    H = H - torch.diag_embed(torch.diagonal(H))

    pruned_edge_mask = prune_edge_mask(H, batch.edge_flag).to(device)
    explained_edge_attr = batch.edge_attr * pruned_edge_mask.view(1, -1).sigmoid()
    explained_edge_attr = explained_edge_attr.squeeze()
    batch.edge_attr = explained_edge_attr
    batch.edge_flag = pruned_edge_mask

    # m = torch.sigmoid(hparams_vec_mask)
    # M = vec_to_symmetric_mask(m)

    logits_masked = functional_call(
        backbone_model,
        params_state,
        (batch,),
        # {"mask_mat": M},
    )

    # ---- supervised SOAP term ----
    y = batch.y.long().view(-1)
    # class_loss = F.cross_entropy(logits_masked, y)
    class_loss = soap_loss_from_logits(logits_masked, y, "label")

    # ---- agreement term: encourage masked prediction to match original prediction ----
    mask_loss = F.cross_entropy(logits_masked, pred_label)

    m = H.sigmoid()
    I = torch.eye(N, device=m.device, dtype=torch.bool)

    coeff_edge_size = 0.05
    # sparse_loss = coeff_edge_size * m.sum() / 10.0
    sparse_loss = coeff_edge_size * m.masked_fill(I, 0.0).sum() / 10.0

    coeff_edge_ent = 0.1
    EPS = 1e-15
    ent = -m * torch.log(m + EPS) - (1.0 - m) * torch.log(1.0 - m + EPS)
    entropy_loss = coeff_edge_ent * ent[~I].mean()
    # ent = -m * torch.log(m + EPS) - (1.0 - m) * torch.log(1.0 - m + EPS)
    # entropy_loss = coeff_edge_ent * ent.mean()

    total_loss = class_loss + 0.3 * mask_loss + sparse_loss + entropy_loss
    # total_loss = class_loss + mask_loss + sparse_loss * 1000 + entropy_loss * 100

    # print("class_loss=", class_loss)
    # print("mask_loss=", mask_loss)
    # print("sparse_loss=", sparse_loss)
    # print("entropy_loss=", entropy_loss)
    # print("total_loss=", total_loss)

    #### only CE
    # y = batch.y.long().view(-1)
    # total_loss = F.cross_entropy(logits_masked, y)

    return total_loss


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def eval_acc_gnn(
    loader,
    backbone_model: IBGNN,
    spec_all,
    params_vec_full: torch.Tensor,
    hparams_vec_mask: torch.Tensor,
    N,
    threshold=None,  # if given (binary only), use y_score[:,1] >= threshold instead of argmax
):
    """
    在整个 loader 上评估 masked GNN 的 loss / acc。
    返回: (avg_loss, avg_acc)
    """
    device = next(backbone_model.parameters()).device

    # mask_prob = torch.sigmoid(hparams_vec_mask.detach())
    # M = vec_to_symmetric_mask(mask_prob).to(device)

    params_state = state_from_vec(backbone_model, spec_all, params_vec_full, device)

    was_training = backbone_model.training
    backbone_model.eval()

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_y = []
    all_scores = []

    # hparams_vec_mask = hparams_vec_mask.reshape(N, N)
    # hparams_vec_mask = (hparams_vec_mask + hparams_vec_mask.T) / 2

    H = hparams_vec_mask.view(N, N)
    H = 0.5 * (H + H.T)
    H = H - torch.diag_embed(torch.diagonal(H))

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ############### refer original code
            pruned_edge_mask = prune_edge_mask(H, batch.edge_flag).to(device)
            explained_edge_attr = batch.edge_attr * pruned_edge_mask.view(1, -1).sigmoid()
            explained_edge_attr = explained_edge_attr.squeeze()
            batch.edge_attr = explained_edge_attr
            batch.edge_flag = pruned_edge_mask

            logits = functional_call(backbone_model, params_state, (batch,))
            y = batch.y.long().view(-1)
            loss = F.cross_entropy(logits, y)

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

            probs = F.softmax(logits, dim=-1)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_y.append(y.cpu().numpy())
            all_scores.append(probs.cpu().numpy())

    backbone_model.train(was_training)

    avg_loss = total_loss / max(total_samples, 1)

    y_true = np.concatenate(all_y)
    y_score = np.concatenate(all_scores)

    if threshold is not None and y_score.shape[1] == 2:
        y_pred = (y_score[:, 1] >= threshold).astype(np.int64)
    else:
        y_pred = np.concatenate(all_preds)

    acc = (y_pred == y_true).mean()
    f1 = f1_score(y_true, y_pred, average="macro")

    try:
        if y_score.shape[1] == 2:
            auc = roc_auc_score(y_true, y_score[:, 1])
        else:
            auc = roc_auc_score(y_true, y_score, multi_class="ovr")
    except:
        auc = 0.5

    try:
        if y_score.shape[1] == 2:
            auprc = average_precision_score(y_true, y_score[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=list(range(y_score.shape[1])))
            auprc = average_precision_score(y_true_bin, y_score, average="macro")
    except ValueError:
        auprc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    f1_each_class = f1_score(y_true, y_pred, average=None)

    return avg_loss, acc, f1, auc, auprc, cm, f1_each_class, y_true, y_score


def get_n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


def save_fig(num_step, result, save_folder, best_test_acc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = np.arange(num_step + 1)

    plt.subplot(1, 2, 1)
    plt.plot(steps, result["train_loss"], label="Train Loss")
    plt.plot(steps, result["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train / Val Loss")

    plt.subplot(1, 2, 2)
    plt.plot(steps, result["val_acc"], label="Val Acc")
    plt.plot(steps, result["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Val / Test Accuracy")

    plt.tight_layout()
    plt.show()

    acc_str = f"{best_test_acc*100:.2f}pct"

    fig_path = os.path.join(save_folder, f"metric_BestTestAcc_at_BestVal_{acc_str}.pdf")
    fig.savefig(fig_path, dpi=500, bbox_inches="tight")
    print("save figure =", fig_path)


def save_fig_cdr(
    num_step, result, save_folder, best_test_f1, best_test_acc, best_test_auc, cm, f1_each
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    steps = np.arange(num_step + 1)

    axes[0].plot(steps, result["train_loss"], label="Train Loss", color="#1f77b4")
    axes[0].plot(steps, result["val_loss"], label="Val Loss", color="#ff7f0e")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(steps, result["val_acc"], label="Val Acc", color="red", linestyle="-")
    axes[1].plot(
        steps, result["test_acc"], label="Test Acc", color="red", linestyle="--", alpha=0.6
    )

    axes[1].plot(steps, result["val_f1"], label="Val F1", color="green", linestyle="-")
    axes[1].plot(
        steps, result["test_f1"], label="Test F1", color="green", linestyle="--", alpha=0.6
    )

    axes[1].plot(steps, result["val_auc"], label="Val AUC", color="blue", linestyle="-")
    axes[1].plot(
        steps, result["test_auc"], label="Test AUC", color="blue", linestyle="--", alpha=0.6
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (0.0 - 1.0)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Performance Metrics Comparison")
    axes[1].legend(loc="lower right", fontsize="small", ncol=2)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    name = f"Best_F1-{best_test_f1:.4f}_Acc-{best_test_acc:.4f}_AUC-{best_test_auc:.4f}.pdf"
    fig_path = os.path.join(save_folder, name)
    fig.savefig(fig_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure saved: {name}")

    ###### save figure
    plt.figure(figsize=(8, 6))
    labels = ["0", "1", "2"] if cm.shape[0] == 3 else [f"Class {i}" for i in range(cm.shape[0])]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Best Test Confusion Matrix\nF1:{best_test_f1:.4f} Acc:{best_test_acc:.4f}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    cm_path = os.path.join(save_folder, f"Confusion_Matrix_Best.png")
    plt.savefig(cm_path)
    plt.close()


def save_fig_bilevel(num_step, result, save_folder, best_test_acc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = np.arange(num_step + 1)

    plt.subplot(1, 2, 1)
    plt.plot(steps, result["train_loss"], label="Train Loss")
    plt.plot(steps, result["upper_loss"], label="Upper Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train / Upper Loss")

    plt.subplot(1, 2, 2)
    plt.plot(steps, result["val_acc"], label="Val Acc")
    plt.plot(steps, result["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Upper / Test Accuracy")

    plt.tight_layout()
    plt.show()

    acc_str = f"{best_test_acc*100:.2f}pct"

    fig_path = os.path.join(save_folder, f"metric_BestTestAcc_at_BestVal_{acc_str}.pdf")
    fig.savefig(fig_path, dpi=500, bbox_inches="tight")
    print("save figure =", fig_path)


def plot_mask_half(save_folder: str, N):
    """
    读取 save_folder/backbone_mask.pt 中保存的 mask_vec，
    还原成 [N, N] 对称矩阵，并用 seaborn.heatmap 画出来。
    """
    ckpt_path = os.path.join(save_folder, "backbone_mask.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[plot_mask] checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "mask" not in ckpt:
        print(f"[plot_mask] 'mask' not found in checkpoint: {ckpt_path}")
        return

    mask_vec = ckpt["mask"].float().view(-1)  # [len_mask]

    M = vec_to_symmetric_mask(torch.sigmoid(mask_vec), N)
    # M = (torch.sigmoid(mask_vec)).reshape(N, N)
    M_np = M.detach().cpu().numpy()

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

    out_path = os.path.join(save_folder, "mask_matrix.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close()
    print("[plot_mask] save mask figure =", out_path)


def plot_mask(save_folder: str, N):
    """
    读取 save_folder/backbone_mask.pt 中保存的 mask_vec，
    还原成 [N, N] 对称矩阵，并用 seaborn.heatmap 画出来。
    """
    ckpt_path = os.path.join(save_folder, "backbone_mask.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[plot_mask] checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "mask" not in ckpt:
        print(f"[plot_mask] 'mask' not found in checkpoint: {ckpt_path}")
        return

    mask_vec = ckpt["mask"].float().view(-1)  # [len_mask]

    # M = vec_to_symmetric_mask(torch.sigmoid(mask_vec))
    M = torch.sigmoid(mask_vec).reshape(N, N)
    M.fill_diagonal_(0.0)
    M_np = M.detach().cpu().numpy()

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

    out_path = os.path.join(save_folder, "mask_matrix.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close()
    print("[plot_mask] save mask figure =", out_path)


def soap_loss_from_logits(logits, labels, batch_idx, soap_loss, tag):
    probs = F.softmax(logits, dim=-1)[:, 1]
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        raise ValueError(
            f"[soap_loss_from_logits] {tag} batch has only one class; "
            "SOAPLOSS requires both classes."
        )
    f_ps = probs[pos_mask]
    f_ns = probs[neg_mask]
    index_s = batch_idx[pos_mask]
    return soap_loss(f_ps, f_ns, index_s)


def lower_loss_gnn_CDR(
    batch,
    backbone_model: IBGNN,
    spec_all,
    params_vec_full: torch.Tensor,
    hparams_vec_mask: torch.Tensor,
    N,
    soap_loss,
    l2_reg: float = 0.0,
) -> torch.Tensor:
    device = next(backbone_model.parameters()).device
    batch = batch.to(device)

    ############### refer original code  (W+W^T)/2
    # hparams_vec_mask = hparams_vec_mask.reshape(N, N)
    # hparams_vec_mask = (hparams_vec_mask + hparams_vec_mask.T) / 2
    H = hparams_vec_mask.view(N, N)
    H = 0.5 * (H + H.T)
    H = H - torch.diag_embed(torch.diagonal(H))

    pruned_edge_mask = prune_edge_mask(H, batch.edge_flag).to(device)
    explained_edge_attr = batch.edge_attr * pruned_edge_mask.view(1, -1).sigmoid()
    explained_edge_attr = explained_edge_attr.squeeze()
    batch.edge_attr = explained_edge_attr
    batch.edge_flag = pruned_edge_mask

    # # mask = hparams_vec_mask.to(device).view(N, N)
    # mask_prob = torch.sigmoid(hparams_vec_mask)  # [len_mask]
    # M = vec_to_symmetric_mask(mask_prob)  # [N, N]

    # print(M)

    params_state = state_from_vec(backbone_model, spec_all, params_vec_full, device)

    logits = functional_call(
        backbone_model,
        params_state,
        (batch,),
    )
    y = batch.y.long().view(-1)

    # 4. Cross Entropy loss
    # loss = F.cross_entropy(logits, y)

    # SOAP Loss (optimize AUPRC, binary CDR)
    if not hasattr(batch, "idx"):
        raise ValueError("[lower_loss_gnn_CDR] batch.idx not found for SOAPLOSS")
    batch_idx = batch.idx.view(-1).to(device)
    loss = soap_loss_from_logits(logits, y, batch_idx, soap_loss, "lower")

    if l2_reg > 0:
        loss = loss + l2_reg * params_vec_full.pow(2).mean()

    return loss


def upper_loss_gnn_CDR(
    batch,
    backbone_model: IBGNN,
    spec_all,
    params_vec_full: torch.Tensor,  # y
    hparams_vec_mask: torch.Tensor,  # m
    N,
    soap_loss,
) -> torch.Tensor:
    """
    用一个 Data batch 来计算 upper-level loss：
      L_upper(y, m) = CE(backbone(batch; y, m), batch.y)
    """
    device = next(backbone_model.parameters()).device
    batch = batch.to(device)

    params_state = state_from_vec(backbone_model, spec_all, params_vec_full, device)

    #####
    with torch.no_grad():
        logits_full = functional_call(
            backbone_model,
            params_state,
            (batch,),
        )
        pred_label = logits_full.argmax(dim=-1)  # [B]

    ############### refer original code
    # hparams_vec_mask = hparams_vec_mask.reshape(N, N)
    # hparams_vec_mask = (hparams_vec_mask + hparams_vec_mask.T) / 2
    H = hparams_vec_mask.view(N, N)
    H = 0.5 * (H + H.T)
    H = H - torch.diag_embed(torch.diagonal(H))

    pruned_edge_mask = prune_edge_mask(H, batch.edge_flag).to(device)
    explained_edge_attr = batch.edge_attr * pruned_edge_mask.view(1, -1).sigmoid()
    explained_edge_attr = explained_edge_attr.squeeze()
    batch.edge_attr = explained_edge_attr
    batch.edge_flag = pruned_edge_mask

    # m = torch.sigmoid(hparams_vec_mask)
    # M = vec_to_symmetric_mask(m)

    logits_masked = functional_call(
        backbone_model,
        params_state,
        (batch,),
        # {"mask_mat": M},
    )

    # ---- supervised SOAP term ----
    y = batch.y.long().view(-1)
    if not hasattr(batch, "idx"):
        raise ValueError("[upper_loss_gnn_CDR] batch.idx not found for SOAPLOSS")
    batch_idx = batch.idx.view(-1).to(device)
    class_loss = soap_loss_from_logits(logits_masked, y, batch_idx, soap_loss, "label")

    # ---- agreement term: encourage masked prediction to match original prediction ----
    mask_loss = F.cross_entropy(logits_masked, pred_label)    

    m = H.sigmoid()
    I = torch.eye(N, device=m.device, dtype=torch.bool)

    coeff_edge_size = 0.05
    # sparse_loss = coeff_edge_size * m.sum() / 10.0
    sparse_loss = coeff_edge_size * m.masked_fill(I, 0.0).sum() / 10.0

    coeff_edge_ent = 0.1
    EPS = 1e-15
    ent = -m * torch.log(m + EPS) - (1.0 - m) * torch.log(1.0 - m + EPS)
    entropy_loss = coeff_edge_ent * ent[~I].mean()
    # ent = -m * torch.log(m + EPS) - (1.0 - m) * torch.log(1.0 - m + EPS)
    # entropy_loss = coeff_edge_ent * ent.mean()

    total_loss = class_loss + 0.3 * mask_loss + sparse_loss + entropy_loss
    # total_loss = class_loss + mask_loss + sparse_loss * 1000 + entropy_loss * 100

    # print("class_loss=", class_loss)
    # print("mask_loss=", mask_loss)
    # print("sparse_loss=", sparse_loss)
    # print("entropy_loss=", entropy_loss)
    # print("total_loss=", total_loss)

    #### only CE
    # y = batch.y.long().view(-1)
    # total_loss = F.cross_entropy(logits_masked, y)

    return total_loss


class SOAPLOSS(nn.Module):
    def __init__(self, threshold, data_length, loss_type='sqh', gamma=0.9, device=None):
        '''
        :param threshold: margin for squared hinge loss
        :param device: torch device (e.g. logits.device); if None, use cuda if available
        '''
        super(SOAPLOSS, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("u_all", torch.zeros(data_length, 1, device=device))
        self.register_buffer("u_pos", torch.zeros(data_length, 1, device=device))
        self.threshold = threshold
        self.loss_type = loss_type
        self.gamma = gamma
        print("The loss type is :", self.loss_type)

    def forward(self, f_ps, f_ns, index_s):
        f_ps = f_ps.view(-1)
        f_ns = f_ns.view(-1)

        vec_dat = torch.cat((f_ps, f_ns), 0)
        mat_data = vec_dat.repeat(len(f_ps), 1)

        f_ps = f_ps.view(-1, 1)

        neg_mask = torch.ones_like(mat_data)
        neg_mask[:, 0:f_ps.size(0)] = 0

        pos_mask = torch.zeros_like(mat_data)
        pos_mask[:, 0:f_ps.size(0)] = 1

        if self.loss_type == 'sqh':
            pos_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * pos_mask
            neg_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * neg_mask

        elif self.loss_type == 'lgs':

            neg_loss = logistic_loss(f_ps, mat_data, self.threshold) * neg_mask
            pos_loss = logistic_loss(f_ps, mat_data, self.threshold) * pos_mask

        elif self.loss_type == 'sgm':

            neg_loss = sigmoid_loss(f_ps, mat_data, self.threshold) * neg_mask
            pos_loss = sigmoid_loss(f_ps, mat_data, self.threshold) * pos_mask

        loss = pos_loss + neg_loss

        if f_ps.size(0) == 1:
            self.u_pos[index_s] = (1 - self.gamma) * self.u_pos[index_s] + self.gamma * (pos_loss.mean())
            self.u_all[index_s] = (1 - self.gamma) * self.u_all[index_s] + self.gamma * (loss.mean())
        else:
            self.u_all[index_s] = (1 - self.gamma) * self.u_all[index_s] + self.gamma * (loss.mean(1, keepdim=True))
            self.u_pos[index_s] = (1 - self.gamma) * self.u_pos[index_s] + self.gamma * (pos_loss.mean(1, keepdim=True))


        p = (self.u_pos[index_s] - (self.u_all[index_s]) * pos_mask) / (self.u_all[index_s] ** 2)

        p.detach_()
        loss = torch.sum(p * loss)

        return loss


def sigmoid_loss(pos, neg, beta=2.0):
    return 1.0 / (1.0 + torch.exp(beta * (pos - neg)))

def logistic_loss(pos, neg, beta = 1):
    return -torch.log(1/(1+torch.exp(-beta * (pos - neg))))