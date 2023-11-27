import torch
import torch.nn as nn
import torch.nn.functional as f
from Layers import GraphConvolution
import numpy as np
import time


class CoxPath(nn.Module):
    def __init__(self, n_feat, n_hid, num_path, dropout, num_clinical):
        super(CoxPath, self).__init__()
        self.dropout = dropout

        self.gcn1 = GraphConvolution(n_feat, n_hid)
        self.gcn2 = GraphConvolution(n_hid, n_hid)
        self.lin1 = nn.Linear(n_hid, 1)
        self.lin2 = nn.Linear(num_path + num_clinical, 1)

        self.activ = nn.Tanh()
        self.BN = nn.BatchNorm1d(num_path + num_clinical)

    def forward(self, x, adj, clinical, out=False):
        x = self.activ(self.gcn1(x, adj))
        x = f.dropout(x, self.dropout, training=self.training)
        x = self.activ(self.gcn2(x, adj))
        x = f.dropout(x, self.dropout, training=self.training)
        x = self.activ(self.lin1(x))
        x = torch.squeeze(x)
        if out:
            t = round(time.time())
            np.save('log/node_' + str(t) + '.npy', x.detach().cpu().numpy())
        x = torch.cat([x, clinical], dim=1)
        # x = self.BN(x)
        x = self.lin2(x)

        return x
