import torch


def C_index(pred, time, event):
    n_sample = len(time)
    time_indicator = R_set(time)
    time_matrix = time_indicator - torch.diag(torch.diag(time_indicator))
    censor_idx = (event == 0).nonzero()
    zeros = torch.zeros(n_sample)
    time_matrix[censor_idx, :] = zeros
    pred_matrix = torch.zeros_like(time_matrix)
    for j in range(n_sample):
        for i in range(n_sample):
            if pred[i] < pred[j]:
                pred_matrix[j, i] = 1
            elif pred[i] == pred[j]:
                pred_matrix[j, i] = 0.5

    concord_matrix = pred_matrix.mul(time_matrix)
    concord = torch.sum(concord_matrix)
    epsilon = torch.sum(time_matrix)
    concordance_index = torch.div(concord, epsilon)
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
    return concordance_index


def Likelihood(pred, time, event):
    n_observed = event.sum(0)
    time_indicator = R_set(time)
    if torch.cuda.is_available():
        time_indicator = time_indicator.cuda()
    risk_set_sum = time_indicator.mm(torch.exp(pred))
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(event)
    cost = (- (sum_diff_in_observed / n_observed)).reshape(-1,)
    return cost


def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return indicator_matrix
