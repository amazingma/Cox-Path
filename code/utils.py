import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch.utils.data as udata
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


def get_map(adj, pathlist):
    adj = adj[adj['0'].isin(pathlist)]
    adj = adj[adj['1'].isin(pathlist)]
    idx = np.array(pathlist, dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.array(adj.values.tolist())
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.str).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(idx.shape[0], idx.shape[0]), dtype=np.float32)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj


def normalize(adj):
    degree = np.array(adj.sum(1))
    d_inv_sq = np.power(degree, -0.5).flatten()
    d_inv_sq[np.isinf(d_inv_sq)] = 0.
    d_mat_inv_sq = sp.diags(d_inv_sq)
    mx = adj.dot(d_mat_inv_sq).transpose().dot(d_mat_inv_sq)
    return mx


def get_split(x, y, num_clinical, f):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y[:, num_clinical + 1], random_state=f)
    x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=len(y_test), stratify=y_train[:, num_clinical + 1], random_state=f)

    time_train = np.array((y_train[:, num_clinical]).tolist())
    sort_train = np.argsort(-time_train)
    x_train = x_train[sort_train]
    y_train = y_train[sort_train]
    time_vali = np.array(y_vali[:, num_clinical].tolist())
    sort_vali = np.argsort(-time_vali)
    x_vali = x_vali[sort_vali]
    y_vali = y_vali[sort_vali]
    time_test = np.array(y_test[:, num_clinical].tolist())
    sort_test = np.argsort(-time_test)
    x_test = x_test[sort_test]
    y_test = y_test[sort_test]

    return x_train, y_train, x_vali, y_vali, x_test, y_test


def getLoader(x, y, batch):
    dataset = udata.TensorDataset(x, y)
    loader = udata.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    return loader


def preprocess(data):
    data = data.groupby(data.index).mean()
    for j in range(0, data.shape[1]):
        col = data.iloc[:, j]
        na_num = np.count_nonzero(col != col)
        if na_num != 0:
            col_val = col[col == col]
            col.iloc[np.isnan(col)] = col_val.mean()
    return data
