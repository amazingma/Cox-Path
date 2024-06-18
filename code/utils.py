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


# 分层{2(round), 2(random), 6(random)}
def get_skf(k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=666)
    return skf
def get_stratify(x, y, l, num_clinical):
    x_train, x_vali, y_train, y_vali = train_test_split(x, y, test_size=l, stratify=y[:, num_clinical + 1], shuffle=True, random_state=6666)
    return x_train, x_vali, y_train, y_vali


# 1(round), 9{1(random), 4}
def get_chunk(k, x, y, f, num_clinical):
    x = torch.chunk(x, k, dim=0)
    y = torch.chunk(y, k, dim=0)
    x_test = x[f]
    y_test = y[f]
    if f == 0:
        x_last = torch.cat(x[1:10], dim=0)
        y_last = torch.cat(y[1:10], dim=0)
    elif f == 9:
        x_last = torch.cat(x[0:9], dim=0)
        y_last = torch.cat(y[0:9], dim=0)
    else:
        x_1 = torch.cat(x[0:f], dim=0)
        x_2 = torch.cat(x[f+1:10], dim=0)
        x_last = torch.cat([x_1, x_2], dim=0)
        y_1 = torch.cat(y[0:f], dim=0)
        y_2 = torch.cat(y[f+1:10], dim=0)
        y_last = torch.cat([y_1, y_2], dim=0)
    x_last = torch.chunk(x_last, 5, dim=0)
    y_last = torch.chunk(y_last, 5, dim=0)
    x_valid = x_last[0]
    y_valid = y_last[0]
    x_train = torch.cat(x_last[1:5], dim=0)
    y_train = torch.cat(y_last[1:5], dim=0)

    time_train = np.array((y_train[:, num_clinical]).tolist())
    sort_train = np.argsort(-time_train)
    x_train = x_train[sort_train]
    y_train = y_train[sort_train]
    time_valid = np.array(y_valid[:, num_clinical].tolist())
    sort_valid = np.argsort(-time_valid)
    x_valid = x_valid[sort_valid]
    y_valid = y_valid[sort_valid]
    time_test = np.array(y_test[:, num_clinical].tolist())
    sort_test = np.argsort(-time_test)
    x_test = x_test[sort_test]
    y_test = y_test[sort_test]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# 1(round), 1, 8
def get_fold(k, x, y, f, num_clinical):
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    x_test, y_test = None, None

    for i in range(k):
        idx = slice(i * fold_size, (i + 1) * fold_size)
        x_part, y_part = x[idx, ::], y[idx, :]
        if f == k-1:
            if i == 0:
                x_test, y_test = x_part, y_part
            elif i == f:
                x_valid, y_valid = x_part, y_part
            elif x_train is None:
                x_train, y_train = x_part, y_part
            else:
                x_train = torch.cat((x_train, x_part), dim=0)
                y_train = torch.cat((y_train, y_part), dim=0)
        else:
            if i == f:
                x_valid, y_valid = x_part, y_part
            elif i == f + 1:
                x_test, y_test = x_part, y_part
            elif x_train is None:
                x_train, y_train = x_part, y_part
            else:
                x_train = torch.cat((x_train, x_part), dim=0)
                y_train = torch.cat((y_train, y_part), dim=0)

    # 标签就是ostime的降序顺序
    time_train = np.array((y_train[:, num_clinical]).tolist())
    sort_train = np.argsort(-time_train)
    x_train = x_train[sort_train]
    y_train = y_train[sort_train]
    time_valid = np.array(y_valid[:, num_clinical].tolist())
    sort_valid = np.argsort(-time_valid)
    x_valid = x_valid[sort_valid]
    y_valid = y_valid[sort_valid]
    time_test = np.array(y_test[:, num_clinical].tolist())
    sort_test = np.argsort(-time_test)
    x_test = x_test[sort_test]
    y_test = y_test[sort_test]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def getLoader(x, y, batch):
    dataset = udata.TensorDataset(x, y)
    loader = udata.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    return loader


def preprocess(data):
    # 重复样本取均值
    data = data.groupby(data.index).mean()
    # 将NA填充为列均值
    for j in range(0, data.shape[1]):
        col = data.iloc[:, j]
        na_num = np.count_nonzero(col != col)
        if na_num != 0:
            col_val = col[col == col]
            col.iloc[np.isnan(col)] = col_val.mean()
    return data
