"""
Author: Zhiheng Zhou

This module defines a Functional connectom（FC）based（Graph Neural Network (GNN) model for fMRI analysis using PyTorch and dgl.

"""

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import random
from torch.nn.parameter import Parameter

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class BGAN(nn.Module):
    def __init__(self, in_dim, out_dim, n_classes, max_neighs=10):
        super(BGAN, self).__init__()
        self.in_dim = in_dim
        self.max_neighs = max_neighs
        self.classify = nn.Linear(out_dim, n_classes)
        self.GATLayer = GATLayer(in_dim, out_dim, max_neighs)
        self.conv = dglnn.GraphConv(in_dim, 1)
        if out_dim < in_dim:
            raise ValueError(
                f"out_dim ({out_dim}) must be >= in_dim ({in_dim}) for CNN kernel."
            )
        out_width = out_dim - in_dim + 1
        self.local_out_dim = (self.max_neighs - 1) * out_width + out_dim
        self.localw = Parameter(th.randn(self.local_out_dim, out_dim))
        self.h_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.ReLU = nn.ReLU(inplace=True)
        self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax(dim=1)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.localw, gain=gain)
        nn.init.xavier_normal_(self.h_proj.weight, gain=gain)

    def get_graphs(self, block):
        g = block

        in_edges = th.tensor([]).to(device)
        out_edges = th.tensor([]).to(device)
        for node in range(len(g.dstnodes())):
            src, dst = g.in_edges(int(node))
            if src.shape[0] == 0:
                src = th.tensor([int(node)], dtype=th.int64)
            src = src.repeat(self.max_neighs)[: self.max_neighs]
            dst = dst.repeat(self.max_neighs)[: self.max_neighs]
            out_edges = th.cat([out_edges, src]).to(device)
            in_edges = th.cat([in_edges, dst]).to(device)
        graph = dgl.graph((out_edges.long(), in_edges.long())).to(device)
        return graph

    def GlobalAttention(self, g, h):
        scores = self.conv(g, h).view(-1)
        globalweight = F.softmax(scores, dim=0).view(-1, 1)
        return globalweight

    def LocalAttention(self, g, h):
        graph = self.get_graphs(g)
        output = self.GATLayer(graph, h)
        updatefeat = self.ReLU(th.mm(output, self.localw) + self.h_proj(h))
        return updatefeat

    def forward(self, g, h):
        """Local Attention: Update features layers"""
        feats = self.LocalAttention(g, h)
        """Global Attention: Calculate weight layers"""
        weight = self.GlobalAttention(g, h)

        """Classifier"""
        updatafeat = weight * feats
        with g.local_scope():
            g.ndata["h"] = updatafeat
            hg = dgl.mean_nodes(g, "h")
        return self.classify(hg)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, max_neighs):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.conv = CNN(in_dim, max_neighs)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = th.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h = self.conv(alpha * nodes.mailbox["z"])
        return {"h": h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata["z"] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")


class CNN(th.nn.Module):
    def __init__(self, in_dim, max_neighs):
        super(CNN, self).__init__()
        self.in_dim = in_dim
        self.max_neighs = max_neighs
        self.convrow = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, self.in_dim)),
            th.nn.BatchNorm2d(num_features=1),
            th.nn.ReLU(),
        )
        self.convcol = th.nn.Sequential(
            th.nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=(self.max_neighs, 1)
            ),
            th.nn.BatchNorm2d(num_features=1),
            th.nn.ReLU(),
        )

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        for layer in self.convrow:
            if isinstance(layer, th.nn.Conv2d):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for layer in self.convcol:
            if isinstance(layer, th.nn.Conv2d):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, feats):
        feats = feats.unsqueeze(1)
        featsrow = self.convrow(feats)

        featscol = self.convcol(feats)

        featsrow = featsrow.reshape(featsrow.size(0), -1)
        featscol = featscol.reshape(featscol.size(0), -1)
        feats = th.cat((featsrow, featscol), dim=1)
        return feats
