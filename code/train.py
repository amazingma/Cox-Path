from __future__ import division
from __future__ import print_function
import DataLoader as myLoader
from GraphPath import GraphPath
from Survival import C_index, Likelihood
from utils import get_map, get_split, getLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import copy
import os

# 自定义路径,组学,超参数
path = "../../data/Cox-Path/BRCA/"
omics_files = ['methy', 'mrna']  # 'methy', 'mrna', 'mutation', 'cnv'
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=66, help='Random seed.')  # 固定(66)
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')  # 固定(100)
parser.add_argument('--out', type=int, default=1, help='Number of out units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')  # 固定(0.5)
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')  # 固定(0.005)
parser.add_argument('--weight_decay', type=float, default=0.1, help='L2 loss on parameters.')  # 固定(0.1)
parser.add_argument('--clinical', type=int, default=6, help='Number of clinical information')
parser.add_argument('--k_fold', type=int, default=50, help='K-fold cross validation')
parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=156, help='batch size.')  # 156

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
rint = np.random.randint(1, 1000, args.k_fold)
print(rint)

# 保存features, labels, adj
# features, pathlist, labels = myLoader.feat_extract(path, omics_files)
# edges = myLoader.omix_info(path)
# adj = get_map(edges, pathlist)
# np.save('log/features.npy', features)
# np.save('log/labels.npy', labels)
# np.save('log/adj.npy', adj)
# 读取features, labels, adj
features = np.load('log/features_2_onlymethy.npy', allow_pickle=True)
labels = np.load('log/labels.npy', allow_pickle=True)
adj = np.load('log/adj.npy', allow_pickle=True)
features = torch.tensor(np.array(features), dtype=torch.float32)
labels = torch.tensor(np.array(labels), dtype=torch.float32)
adj = torch.tensor(np.array(adj), dtype=torch.float32)
if args.cuda:
    adj = adj.cuda()


cindex_list = []
def k_fold(k, x, y, epo):
    for f in rint:
        model = GraphPath(n_feat=features.shape[2], n_hid=args.hidden, num_path=features.shape[1], dropout=args.dropout, num_clinical=args.clinical)
        # model = nn.DataParallel(model, device_ids=[0])
        opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        x_train, y_train, x_valid, y_valid, x_test, y_test = get_split(x, y, args.clinical, f)
        loader = getLoader(x_train, y_train, batch=args.batch)

        loss_min = 10.0
        cindex_max = 0.5
        model_best = None
        counter = 0
        for e in range(epo):
            # Early Stopping
            if (e - counter) > 10:  # 固定(10)
                # PATH = 'model/model_' + str(r+1) + '_' + str(f+1) + '.pt'
                # torch.save(model_best, PATH)
                # model_test = torch.load(PATH)
                break
            for bat, (x_bat, y_bat) in enumerate(loader):
                loss_vali, cindex_vali = train(e, x_bat, y_bat, x_valid, y_valid, model, opt)
                if loss_vali < loss_min or cindex_vali > cindex_max:
                    counter = e
                    loss_min = loss_vali
                    cindex_max = cindex_vali
                    model_best = copy.deepcopy(model)

        loss_test, cindex_test = test(model_best, x_test, y_test, save=True)
        print('\033[1;31mTest\033[0m Loss = ' + format(loss_test.item(), '.4f') + ', C-index = ' + format(cindex_test.item(), '.4f'))
        cindex_list.append(cindex_test)
    return


def train(epo, x_train, y_train, x_valid, y_valid, model, opt):
    # torch.cuda.empty_cache()
    if args.cuda:
        x_train = x_train.cuda()
        y_train = y_train.cuda()
    # 临床信息
    clinical_train = y_train[:, :args.clinical]
    os_train = y_train[:, args.clinical].unsqueeze(1)
    event_train = y_train[:, args.clinical + 1].unsqueeze(1)

    model.train()
    opt.zero_grad()
    pred_train = model(x_train, adj, clinical_train)
    loss_train = Likelihood(pred_train, os_train, event_train)
    cindex_train = C_index(pred_train, os_train, event_train)
    loss_train.backward()
    opt.step()

    if not args.fastmode:
        loss_vali, cindex_vali = test(model, x_valid, y_valid)
        # print('Epoch ' + str(epo) + '\033[1;32m Train\033[0m Loss = ' + format(loss_train.item(), '.4f') + ', C-index = ' + format(cindex_train.item(), '.4f')
        #       + ';\033[1;35m Validation\033[0m Loss = ' + format(loss_vali.item(), '.4f') + ', C-index = ' + format(cindex_vali.item(), '.4f'))
        return loss_vali, cindex_vali


def test(model_t, x_test, y_test, save=False):
    # torch.cuda.empty_cache()
    if args.cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
    # 临床信息
    clinical_test = y_test[:, :args.clinical]
    os_test = y_test[:, args.clinical].unsqueeze(1)
    event_test = y_test[:, args.clinical + 1].unsqueeze(1)

    model_t.eval()
    pred_test = model_t(x_test, adj, clinical_test)
    loss_test = Likelihood(pred_test, os_test, event_test)
    cindex_test = C_index(pred_test, os_test, event_test)

    # if save:
    #     np.save('log/' + str(cindex_test) + '_Y_test.npy', os_test)
    #     np.save('log/' + str(cindex_test) + '_p_test.npy', pred_test)
    return loss_test, cindex_test


# Training
t_start = time.time()
k_fold(args.k_fold, features, labels, args.epoch)
print('The Final C-index Mean Value of 50 Times Cross Validation is ' + format(np.mean(cindex_list), '.6f') + '+-' + format(np.std(cindex_list), '.6f'))
print('Total time elapsed: {:.4f}s'.format(time.time() - t_start))