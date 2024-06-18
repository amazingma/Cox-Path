import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch.utils.data as udata
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


def get_map(adj, pathlist):
    # 去除无效节点
    adj = adj[adj['0'].isin(pathlist)]
    adj = adj[adj['1'].isin(pathlist)]
    idx = np.array(pathlist, dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.array(adj.values.tolist())
    # 将边节点ko号换成idx
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.str).reshape(edges_unordered.shape)
    # 根据edges确定每个1的填充位置(data, (row, col))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(idx.shape[0], idx.shape[0]), dtype=np.float32)
    # 将有向图的邻接矩阵扩充为对称矩阵
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 加单位矩阵并归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj


def normalize(adj):
    degree = np.array(adj.sum(1))
    d_inv_sq = np.power(degree, -0.5).flatten()
    # 将degree=0行的r_inv置为0
    d_inv_sq[np.isinf(d_inv_sq)] = 0.
    # 对角矩阵
    d_mat_inv_sq = sp.diags(d_inv_sq)
    # tocoo/tocsr/tocsc
    mx = adj.dot(d_mat_inv_sq).transpose().dot(d_mat_inv_sq)
    return mx


# 分层{2(random), 2(random), 6(random)}
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


# 甲基化预处理
def methyProp():
    or_methy = pd.read_csv("E:/Data/Cox-Path/BRCA/or_methy.csv")
    # 删除含有NA的行(CpG位点)
    or_methy = or_methy.dropna(axis=0, how='any')
    # 每个CpG位点只保留第一个对应的基因
    or_methy = or_methy.replace('(;)(.*)', '', regex=True)
    # 同一基因的多个CpG位点取均值
    or_methy = or_methy.groupby(or_methy.iloc[:, -1]).mean()
    or_methy.to_csv("E:/Data/Cox-Path/BRCA/prop_methy.csv", sep=',', index=True, header=True)
    return


# 拷贝数预处理
def cnvProp():
    or_cnv = pd.read_csv("E:/Data/Cox-Path/OV/or_cnv.csv")
    # 先删除全为NA的列(样本)
    or_cnv = or_cnv.dropna(axis=1, how='all')
    # 后删除NA超过20%的行(基因)
    t = round(or_cnv.shape[1]*0.8)
    or_cnv = or_cnv.dropna(thresh=t)
    # 将NA替换为0
    or_cnv.fillna(0, inplace=True)
    or_cnv.to_csv("E:/Data/Cox-Path/OV/prop_cnv.csv", sep=',', index=False, header=True)
    return


# 表达预处理
def mrnaProp():
    or_mrna = pd.read_csv("E:/Data/Cox-Path/BRCA/or_mrna.csv")
    or_mrna.fillna(0, inplace=True)
    or_mrna = or_mrna.groupby(or_mrna.iloc[:, -2]).mean()
    or_mrna.to_csv("E:/Data/Cox-Path/BRCA/prop_mrna.csv", sep=',', index=True, header=True)
    return


# cg编号-GeneName
def cgMapping():
    cgMap = pd.read_csv("E:/Data/Cox-Path/HNSC/Methyl450_hg38_GDC.csv")
    cg_list = pd.read_csv("E:/Data/Cox-Path/HNSC/list.csv", header=None)
    cg_list = cg_list.iloc[:, 0].values.tolist()
    result = cgMap[cgMap['id'].isin(cg_list)]
    result.to_csv("E:/Data/Cox-Path/HNSC/result.csv", sep=',', index=True, header=False)
    return


# ENSG编号-GeneName
def probeMapping():
    probeMap = pd.read_csv("E:/Data/Cox-Path/OV/probeMap.csv")
    ENSG_list = pd.read_csv("E:/Data/Cox-Path/OV/list_all.csv", header=None)
    ENSG_list = ENSG_list.iloc[:, 0].values.tolist()
    # 取交集
    result = probeMap[probeMap['id'].isin(ENSG_list)]
    result.to_csv("E:/Data/Cox-Path/OV/result.csv", sep=',', index=False, header=False)
    ENSG_inlist = result.iloc[:, 0].values.tolist()
    prop_cnv = pd.read_csv("E:/Data/Cox-Path/OV/prop_cnv.csv")
    # 取交集, id为列名
    prop_cnv = prop_cnv[prop_cnv['id'].isin(ENSG_inlist)]
    prop_cnv.to_csv("E:/Data/Cox-Path/OV/map_cnv.csv", sep=',', index=False, header=True)
    return


# 转置
def transposition():
    prop_methy = pd.read_csv("E:/Data/Cox-Path/BRCA/prop_methy.csv")
    tf_methy = pd.DataFrame(prop_methy.values.T, index=prop_methy.columns, columns=prop_methy.index)
    # 删除全为0的列
    tf_methy = tf_methy.loc[:, (tf_methy != 0).all(axis=0)]
    tf_methy.to_csv("E:/Data/Cox-Path/BRCA/tf_methy.csv", sep=',', index=True, header=False)

    prop_mrna = pd.read_csv("E:/Data/Cox-Path/BRCA/prop_mrna.csv")
    tf_mrna = pd.DataFrame(prop_mrna.values.T, index=prop_mrna.columns, columns=prop_mrna.index)
    # 删除全为0的列
    tf_mrna = tf_mrna.loc[:, (tf_mrna != 0).all(axis=0)]
    tf_mrna.to_csv("E:/Data/Cox-Path/BRCA/tf_mrna.csv", sep=',', index=True, header=False)

    map_cnv = pd.read_csv("E:/Data/Cox-Path/OV/map_cnv.csv")
    tf_cnv = pd.DataFrame(map_cnv.values.T, index=map_cnv.columns, columns=map_cnv.index)
    # 删除全为0的列
    tf_cnv = tf_cnv.loc[:, (tf_cnv != 0).any(axis=0)]
    tf_cnv.to_csv("E:/Data/Cox-Path/OV/tf_cnv.csv", sep=',', index=True, header=False)
    return


def sdFilter():
    # or_methy = pd.read_csv("E:/Data/Cox-Path/BRCA/tf_methy.csv", index_col=0)
    # index = or_methy.index
    # columns = or_methy.columns
    # vf = VarianceThreshold(threshold=0.0045)
    # vf_methy = vf.fit_transform(or_methy)
    # select_index = vf.get_support(indices=True)
    # columns = columns[select_index]
    # fi_methy = pd.DataFrame(vf_methy)
    # fi_methy.index = index
    # fi_methy.columns = columns
    # fi_methy.to_csv("E:/Data/Cox-Path/BRCA/fi_methy.csv", sep=',', index=True, header=True)

    or_mrna = pd.read_csv("E:/Data/Cox-Path/BRCA/tf_mrna.csv", index_col=0)
    index = or_mrna.index
    columns = or_mrna.columns
    # 先归一化
    minmax = MinMaxScaler()
    mm_mrna = minmax.fit_transform(or_mrna)
    mm_mrna = pd.DataFrame(mm_mrna)
    # 再筛选
    vf = VarianceThreshold(threshold=0.006)
    vf_mrna = vf.fit_transform(mm_mrna)
    select_index = vf.get_support(indices=True)
    columns = columns[select_index]
    fi_mrna = pd.DataFrame(vf_mrna)
    fi_mrna.index = index
    fi_mrna.columns = columns
    fi_mrna.to_csv("E:/Data/Cox-Path/BRCA/fi_mrna.csv", sep=',', index=True, header=True)

    or_cnv = pd.read_csv("E:/Data/Cox-Path/OV/tf_cnv.csv", index_col=0)
    index = or_cnv.index
    columns = or_cnv.columns
    # 先归一化
    minmax = MinMaxScaler()
    mm_cnv = minmax.fit_transform(or_cnv)
    mm_cnv = pd.DataFrame(mm_cnv)
    # 再筛选
    vf = VarianceThreshold(threshold=0.025)
    vf_cnv = vf.fit_transform(mm_cnv)
    select_index = vf.get_support(indices=True)
    columns = columns[select_index]
    fi_cnv = pd.DataFrame(vf_cnv)
    fi_cnv.index = index
    fi_cnv.columns = columns
    fi_cnv.to_csv("E:/Data/Cox-Path/OV/fi_cnv.csv", sep=',', index=True, header=True)
    return


# 读取arff文件
def read_arff(file):
    with open(file, encoding='utf-8') as data:
        header = []
        for line in data:
            if line.startswith('@attribute'):
                header.append(line.split()[1])
            elif line.startswith('@data'):
                break
        data_df = pd.read_csv(data, header=None)
        data_df.columns = header
    return data_df
