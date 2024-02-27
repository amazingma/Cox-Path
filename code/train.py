from __future__ import division
from __future__ import print_function
import DataLoader as myLoader
from CoxPath import CoxPath
from Survival import C_index, Likelihood
from utils import get_map, get_split, getLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import random
import copy
import os

path = "data/BRCA/"
omics_files = ['methy', 'mrna', 'cnv', 'mutation']
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=66, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')
parser.add_argument('--out', type=int, default=1, help='Number of out units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.1, help='L2 loss on parameters.')
parser.add_argument('--clinical', type=int, default=6, help='Number of clinical information')
parser.add_argument('--k_fold', type=int, default=5, help='K-fold cross validation')
parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=156, help='batch size.')

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# features, pathlist, labels = myLoader.feat_extract(path, omics_files)
# edges = myLoader.omix_info(path)
# adj = get_map(edges, pathlist)
# np.save('log/features.npy', features)
# np.save('log/labels.npy', labels)
# np.save('log/adj.npy', adj)

features = np.load('log/features.npy', allow_pickle=True)
labels = np.load('log/labels.npy', allow_pickle=True)
adj = np.load('log/adj.npy', allow_pickle=True)
features = torch.tensor(np.array(features), dtype=torch.float32)
labels = torch.tensor(np.array(labels), dtype=torch.float32)
adj = torch.tensor(np.array(adj), dtype=torch.float32)
if args.cuda:
    adj = adj.cuda()


def k_fold(r, k, x, y, epo):
    index = [i for i in range(len(y))]
    random.shuffle(index)
    x = x[index]
    y = y[index]

    cindex_tot = 0.
    for f in range(k):
        model = CoxPath(n_feat=features.shape[2], n_hid=args.hidden, num_path=features.shape[1], dropout=args.dropout, num_clinical=args.clinical)
        model = nn.DataParallel(model, device_ids=[0])
        opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        x_train, y_train, x_valid, y_valid, x_test, y_test = get_split(x, y, args.clinical)
        loader = getLoader(x_train, y_train, batch=args.batch)

        loss_min = 10.0
        cindex_max = 0.5
        model_best = None
        counter = 0
        for e in range(epo):
            # Early Stopping
            if (e - counter) > 10:
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
        cindex_tot += cindex_test.item()
    cindex_mean = cindex_tot / k
    print('Optimization finished! C-index mean value of 5-f is ' + str(cindex_mean))
    return cindex_mean


def train(epo, x_train, y_train, x_valid, y_valid, model, opt):
    # torch.cuda.empty_cache()
    if args.cuda:
        x_train = x_train.cuda()
        y_train = y_train.cuda()
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
    clinical_test = y_test[:, :args.clinical]
    os_test = y_test[:, args.clinical].unsqueeze(1)
    event_test = y_test[:, args.clinical + 1].unsqueeze(1)

    model_t.eval()
    with torch.no_grad():
        pred_test = model_t(x_test, adj, clinical_test)
        loss_test = Likelihood(pred_test, os_test, event_test)
        cindex_test = C_index(pred_test, os_test, event_test)

    # if save:
    #     np.save('log/' + str(cindex_test) + '_Y_test.npy', os_test)
    #     np.save('log/' + str(cindex_test) + '_p_test.npy', pred_test)
    return loss_test, cindex_test


# Training
cindex_final = 0.0
t_start = time.time()
for rep in range(10):
    print('<-- The ' + str(rep + 1) + 'th time cross validation -->')
    cindex_r = k_fold(rep, args.k_fold, features, labels, args.epoch)
    cindex_final += cindex_r
print('Total time elapsed: {:.4f}s'.format(time.time() - t_start))
print('The Final C-index Mean Value of 10 Times Cross Validation is ' + str(cindex_final / 10))
