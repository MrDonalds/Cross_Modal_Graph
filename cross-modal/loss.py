import numpy as np
import torch
import scipy.spatial


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim


def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta):
    term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + \
            ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()

    cos = lambda x, y: x.mm(y.t()) / (
        (x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.

    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()

    # 原始的第二个 loss 函数
    term21 = ((1 + torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1 + torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23

    # # 统一模态的 loss 函数
    #     theta = cos(view1_feature+view2_feature, view1_feature+view2_feature)
    #     Sim = calc_label_sim(labels_1+labels_2, labels_1+labels_2).float()
    #     term2 = ((1+torch.exp(theta)).log() - Sim * theta).mean()

    #     term3 = ((view1_feature - view2_feature)**2).sum(1).sqrt().mean()

    im_loss = alpha * term1 + beta * term2  # + gama * term3

    return im_loss


def fx_calc_map_label(image, text, label, k = 0, dist_method='COS'):
#     print(image, text)
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)