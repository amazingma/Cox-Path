import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features, hid_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.hid_features = hid_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, hid_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hid_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        x = torch.einsum('npk,kh->nph', input, self.weight)
        output = torch.einsum('pp,nph->nph', adj, x)
        if self.bias is not None:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.hid_features) + ')'
