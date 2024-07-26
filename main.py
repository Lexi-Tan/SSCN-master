#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/7 14:16
# @Author  : 我的名字
# @File    : PU_PC.py
# @Description : 这个函数是用来balabalabala自己写

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:03:01 2023

@author: Laura
"""

import argparse
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
import random
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from sklearn.decomposition import PCA
from collections import Counter
import copy
import torch.nn as nn
from dataprocessing import padWithZeros
from itertools import cycle


root = os.getcwd()



parser = argparse.ArgumentParser(description="Few Shot Multi-task Classification")
parser.add_argument("-t", "--train_num", type=int, default=200)
parser.add_argument("-f", "--few_num", type=int, default=20)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-e", "--epoches", type=int, default=200)
parser.add_argument("-p", "--patch_size", type=int, default=7)
parser.add_argument("-n", "--numComponents", type=int, default=104)

args = parser.parse_args(args=[])
# Hyper Parameters
patch_size = args.patch_size
train_num = args.train_num
few_num = args.few_num
epoches = args.epoches
numComponents = args.numComponents

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Task(data, label, shot_num, query_num):
    num_classes = int(np.max(label) + 1)
    support_datas = np.empty([num_classes * shot_num, 7, 7, 104])
    support_labels = np.empty([num_classes * shot_num])
    query_datas = np.empty([num_classes * query_num, 7, 7, 104])
    query_labels = np.empty([num_classes * query_num])
    for c in range(num_classes):
        temp = data[c * 200:(c + 1) * 200, :, :, :]
        support_datas[c * shot_num:(c + 1) * shot_num, :, :, :] = temp[:shot_num, :, :, :]
        support_labels[c * shot_num:(c + 1) * shot_num] = c
        query_datas[c * query_num:(c + 1) * query_num, :, :, :] = temp[shot_num:shot_num + query_num, :, :, :]
        query_labels[c * query_num:(c + 1) * query_num] = c

    return torch.tensor(support_datas), torch.tensor(support_labels), torch.tensor(query_datas), torch.tensor(
        query_labels)


class TripletLoss(nn.Module):

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):

        n = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # loss = F.relu(dist_ap - dist_an + self.margin)
        return loss


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


def estimate_cov(examples, rowvar=False, inplace=False):
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()


def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def build_class_reps_and_covariance_estimates(context_features, context_labels):
    class_representations = {}
    class_precision_matrices = {}
    task_covariance_estimate = estimate_cov(context_features)
    for c in torch.unique(context_labels):
        # filter out feature vectors which have class c
        class_mask = torch.eq(context_labels, c)
        class_mask_indices = torch.nonzero(class_mask)
        class_features = torch.index_select(context_features, 0, extract_class_indices(context_labels, c))
        # class_features = torch.index_select(context_features, 0, torch.reshape(class_mask_indices, (-1,)))
        # mean pooling examples to form class means
        class_rep = mean_pooling(class_features)
        # updating the class representations dictionary with the mean pooled representation
        class_representations[c.item()] = class_rep
        lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
        class_precision_matrices[c.item()] = torch.inverse(
            (lambda_k_tau * estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
            + torch.eye(class_features.size(1), class_features.size(1)))
    return class_representations, class_precision_matrices


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


def domain_metric(context_features, context_labels, target_features):
    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(context_features,
                                                                                                context_labels)
    # grabbing the number of classes and query examples for easier use later in the function
    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))
    number_of_classes = class_means.size(0)
    number_of_targets = target_features.size(0)

    repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    return split_first_dim_linear(sample_logits, [target_features.shape[0]])


def nnclassifier1(ans, feature_samples, feature_labels):
    query_features = torch.tensor(np.array(ans))
    support_feature = torch.tensor(np.array(feature_samples))
    support_labels = torch.tensor(np.array(feature_labels))
    sample_logits = domain_metric(support_feature, support_labels, query_features)
    predict_labels = torch.argmax(sample_logits, dim=1)
    return predict_labels.detach().numpy()


def get_target_samples(f, label, class_num):
    train = {}
    test = {}
    for i in range(class_num):
        mask = (label == i)
        indices = torch.nonzero(mask)
        train[i] = indices[:1]
        test[i] = indices[:]
    train_indices = []
    test_indices = []
    for i in range(class_num):
        train_indices += train[i]
        test_indices += test[i]
    support_fea = torch.zeros([len(train_indices), f.shape[1]])
    support_label = torch.zeros([len(train_indices)])
    query_fea = torch.zeros([len(test_indices), f.shape[1]])
    query_label = torch.zeros([len(test_indices)])
    for i in range(len(train_indices)):
        support_fea[i, :] = f[train_indices[i], :]
        support_label[i] = label[train_indices[i]]
    for i in range(len(test_indices)):
        query_fea[i:] = f[test_indices[i], :]
        query_label[i] = label[test_indices[i]]

    return support_fea, support_label, query_fea, query_label


def anchor_feature(train_loader3, train_loader4, device, encoder, decoder, encoder_optimizier, decoder_optimizier):
    ref_samples1 = []
    ref_labels1 = []
    ref_samples2 = []
    ref_labels2 = []

    for indx, data in enumerate(zip(cycle(train_loader4), train_loader3)):
        image3 = data[0][0].to(device)
        label3 = data[0][1].to(device)

        image4 = data[1][0].to(device)  # imagex为PC
        label4 = data[1][1].to(device)

        encoder_optimizier.zero_grad()
        decoder_optimizier.zero_grad()
        with torch.no_grad():  # 在测试的时候必须加上这行和下一行代码，否则预测会出问题，这里是防止还有梯度更新这些，而且如果不加这个，后面又没有进行梯度更新的话，可能会报显存不够用的错误，我怀疑是数据没有被清理
            encoder.eval()
            decoder.eval()  # 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！
            fea3, feas3, fea4, feas4, f3, f4, c3, c4 = encoder(image3, image4)
            d3, d4, ds3, ds4 = decoder(fea3, feas3, fea4, feas4)
            f_3 = torch.cat((f3, ds3), 1)
            f_4 = torch.cat((f4, ds4), 1)
            # f_3 = f3
            # f_4 = f4
            ref_samples1.extend(f_3.cpu().numpy())
            ref_labels1.extend(label3.cpu().numpy())
            ref_samples2.extend(f_4.cpu().numpy())
            ref_labels2.extend(label4.cpu().numpy())
    return ref_samples1, ref_samples2, ref_labels1, ref_labels2


def test(X1, X2, patch_size, device, encoder, decoder, encoder_optimizier, decoder_optimizier):
    shapeor1 = X1.shape
    shapeor2 = X2.shape
    margin = int((patch_size - 1) / 2)
    zeroX1 = padWithZeros(X1, margin)
    zeroX2 = padWithZeros(X2, margin)
    r_max = np.max([shapeor1[0], shapeor2[0]])
    c_max = np.max([shapeor1[1], shapeor2[1]])
    ans1 = []
    ans2 = []
    fea1 = []
    fea2 = []
    for r in range(margin, r_max + margin):
        for c in range(margin, c_max + margin):
            if (r < (shapeor1[0] + margin)) & (c < (shapeor1[1] + margin)):
                patch1 = zeroX1[int(r - margin):int(r + margin + 1), int(c - margin):int(c + margin + 1), :]
                patch1 = torch.FloatTensor(patch1)
                patch1 = patch1.unsqueeze(0)
                image3 = patch1.unsqueeze(0).to(device)
            if (r < (shapeor2[0] + margin)) & (c < (shapeor2[1] + margin)):
                patch2 = zeroX2[int(r - margin):int(r + margin + 1), int(c - margin):int(c + margin + 1), :]
                patch2 = torch.FloatTensor(patch2)
                patch2 = patch2.unsqueeze(0)
                image4 = patch2.unsqueeze(0).to(device)
            encoder_optimizier.zero_grad()
            decoder_optimizier.zero_grad()
            with torch.no_grad():  # 在测试的时候必须加上这行和下一行代码，否则预测会出问题，这里是防止还有梯度更新这些，而且如果不加这个，后面又没有进行梯度更新的话，可能会报显存不够用的错误，我怀疑是数据没有被清理
                encoder.eval()  # 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！
                decoder.eval()
                fea3, feas3, fea4, feas4, f3, f4, c3, c4 = encoder(image3, image4)
                d3, d4, ds3, ds4 = decoder(fea3, feas3, fea4, feas4)
                f_3 = torch.cat((f3, ds3), 1)
                f_4 = torch.cat((f4, ds4), 1)
            if (r < (shapeor1[0] + margin)) & (c < (shapeor1[1] + margin)):
                # 有标签样本
                ans1.extend(c3.cpu().numpy())
                fea1.extend(f_3.cpu().numpy())
            if (r < (shapeor2[0] + margin)) & (c < (shapeor2[1] + margin)):
                # 无标签样本
                ans2.extend(c4.cpu().numpy())
                fea2.extend(f_4.cpu().numpy())
    return ans1, ans2, fea1, fea2

def accuracycompute(ref_samples1, ref_samples2, ref_labels1, ref_labels2, ans1, ans2, fea1, fea2, y1, y2, cla_Labels1,
                    cla_Labels2):
    '''------------------labeled dataset--------------------'''
    label_x1 = nnclassifier1(fea1, ref_samples1, ref_labels1)
    label_x2 = nnclassifier1(fea2, ref_samples2, ref_labels2)
    shapeor1 = y1.shape
    shapeor2 = y2.shape
    true_y1 = []
    true_y2 = []
    for i in range(shapeor1[0]):
        for j in range(shapeor1[1]):
            true_y1.append(y1[i, j])
    for i in range(shapeor2[0]):
        for j in range(shapeor2[1]):
            true_y2.append(y2[i, j])
    label1 = []
    label2 = []
    for i in range(len(true_y1)):
        if true_y1[i] != 0:
            label1.append(label_x1[i])
    for i in range(len(true_y2)):
        if true_y2[i] != 0:
            label2.append(label_x2[i])
    label1 = np.array(label1)
    label2 = np.array(label2)

    from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score

    kappa1 = cohen_kappa_score(cla_Labels1, label1)
    OA1 = accuracy_score(cla_Labels1, label1)
    classification1 = classification_report(cla_Labels1, label1, digits=4)

    '''------------------few-shot dataset--------------------'''

    kappa2 = cohen_kappa_score(cla_Labels2, label2)
    OA2 = accuracy_score(cla_Labels2, label2)
    classification2 = classification_report(cla_Labels2, label2, digits=4)
    return label_x1, label_x2, OA1, OA2, kappa1, kappa2, classification1, classification2

def dataload():
    X1 = sio.loadmat(root + '//denoiseHSI/PaviaU.mat')['paviau']
    y_label1 = sio.loadmat(root + '//denoiseHSI//PaviaU_gt.mat')['paviaU_gt']
    # few-shot task
    X2 = sio.loadmat(root + '//denoiseHSI//paviac.mat')['paviac']
    y_label2 = sio.loadmat(root + '//denoiseHSI//Paviac_gt.mat')['pavia_gt']

    return X1, X2, y_label1, y_label2, root1


def padWithZeros(X, margin):
    if len(np.shape(X)) == 3:
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    if len(np.shape(X)) == 2:
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset] = X
    return newX


def createImageCubes(X, y, cla_flag, patch_size):
    margin = int((patch_size - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin)
    zeroPaddedy = padWithZeros(y, margin)
    # split patches
    patchesLocation = np.zeros((sum(cla_flag), 2))
    patchesLabels = np.zeros((sum(cla_flag)))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if zeroPaddedy[r, c] != 0:
                patchesLocation[patchIndex, 0] = r
                patchesLocation[patchIndex, 1] = c
                patchesLabels[patchIndex] = zeroPaddedy[r, c]
                patchIndex = patchIndex + 1
    patchesLabels = patchesLabels - 1
    return patchesLocation, patchesLabels.astype("int")


def AugData_split(X, cla_flag, num, Location, Labels, patch_size):
    margin = int((patch_size - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin)
    train_Data = np.zeros((len(cla_flag) * num, patch_size, patch_size, X.shape[2]))
    train_labels = np.zeros((len(cla_flag) * num))
    test_Location = []
    test_Labels = []

    mmm = 0
    for i in range(len(cla_flag)):
        location = Location[Labels == i, :]
        labels = Labels[Labels == i]
        random.seed(100)
        rand_index = np.array(random.sample(range(0, cla_flag[i]), cla_flag[i]))
        if cla_flag[i] > num * 0.8:
            train_num = num
        else:
            train_num = int(cla_flag[i] * 0.8)
        train_index = rand_index[:train_num]
        test_index = rand_index[train_num:]
        for k in range(train_num):
            aixs = location[train_index[k], :]
            train_Data[mmm, :, :, :] = zeroPaddedX[int(aixs[0] - margin):int(aixs[0] + margin + 1),
                                       int(aixs[1] - margin):int(aixs[1] + margin + 1), :]
            train_labels[mmm] = labels[train_index[k]]
            mmm += 1
        test_Location.extend(location[test_index])
        test_Labels.extend(labels[test_index])
    test_Location = np.array(test_Location)
    test_Labels = np.array(test_Labels)
    return train_Data, train_labels, test_Location, test_Labels


def Data_split(X, y, S, few_num, patch_size):
    margin = int((patch_size - 1) / 2)
    hsi_number = len(np.unique(y)) - 1  # 高光谱数据地物的种类
    zeroPaddedX = padWithZeros(X)
    zeroPaddedy = padWithZeros(y)
    zeroPaddedS = padWithZeros(S)
    position_y = [[] for _ in range(hsi_number)]
    distribution_s = [[] for _ in range(hsi_number)]
    y_s = [[] for _ in range(hsi_number)]
    Xtrain = []
    ytrain = []
    # 找出所有的labelled 位置
    for i in range(zeroPaddedy.shape[0]):
        for j in range(zeroPaddedy.shape[1]):
            if zeroPaddedy[i, j] != 0:
                label = int(zeroPaddedy[i, j])
                position_y[label - 1].append(([i, j]))
                distribution_s[label - 1].append(int(zeroPaddedS[i, j]))
    pos_sel = [[] for _ in range(hsi_number)]
    for i in range(len(position_y)):
        count = Counter(distribution_s[i])
        distribution_s[i].sort(key=lambda x: count[x], reverse=True)
        num_s = list(set(distribution_s[i]))
        pro = [np.ceil(count[num_s[k]] / len(distribution_s[i]) * few_num) for k in range(len(num_s))]
        label = [[num_s[k]] * int(pro[k]) for k in range(len(num_s))]
        label_s = [b for a in label for b in a]
        indx = 0
        random.seed(i)
        random.shuffle(position_y[i])
        for pos in position_y[i]:
            if indx < few_num:
                s = zeroPaddedS[pos[0], pos[1]]
                if s in label_s:
                    label_s.remove(s)
                    patch = zeroPaddedX[pos[0] - margin:pos[0] + margin + 1, pos[1] - margin:pos[1] + margin + 1]
                    label = int(zeroPaddedy[pos[0], pos[1]])
                    y_s[i].append(int(zeroPaddedS[pos[0], pos[1]]))
                    pos_sel[i].append(pos)
                    Xtrain.append(patch)
                    ytrain.append(label)
                    indx += 1
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    return Xtrain, ytrain, pos_sel, y_s


class Train(Dataset):
    def __init__(self, Xtrain, ytrain):
        self.len = len(ytrain)
        self.x_data = torch.FloatTensor(Xtrain)
        self.x_data = self.x_data.unsqueeze(1)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class Train1(Dataset):
    def __init__(self, Xtrain, ytrain):
        self.ytrain = Xtrain
        self.x_data = torch.FloatTensor(Xtrain)
        self.x_data = self.x_data.unsqueeze(1)
        self.y_data = torch.FloatTensor(Xtrain)
        self.y_data = self.y_data.unsqueeze(1)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return len(self.ytrain)


""" Testing dataset"""


class Test(Dataset):
    def __init__(self, Xtest, ytest):
        self.len = len(ytest)
        self.x_data = torch.FloatTensor(Xtest)
        self.x_data = self.x_data.unsqueeze(1)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class Test1(Dataset):
    def __init__(self, Xtest, ytest):
        self.ytest = Xtest
        self.x_data = torch.FloatTensor(Xtest)
        self.x_data = self.x_data.unsqueeze(1)
        self.y_data = torch.FloatTensor(Xtest)
        self.y_data = self.y_data.unsqueeze(1)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return len(self.ytest)


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def data_augment(Xtrain1_super, ytrain1_super, Xtrain2_few, ytrain2_few):
    if Xtrain1_super.shape[0] > Xtrain2_few.shape[0]:
        X1 = copy.deepcopy(Xtrain1_super)
        X2 = copy.deepcopy(Xtrain2_few)
        y1 = copy.deepcopy(ytrain1_super)
        y2 = copy.deepcopy(ytrain2_few)
    else:
        X2 = copy.deepcopy(Xtrain1_super)
        X1 = copy.deepcopy(Xtrain2_few)
        y2 = copy.deepcopy(ytrain1_super)
        y1 = copy.deepcopy(ytrain2_few)
    quotient, remainder = divmod(X1.shape[0], X2.shape[0])
    # 对数组进行数据增强
    Xtrain2 = []
    ytrain2 = []
    for i in range(quotient):
        Xtrain2.extend(X2)
        ytrain2.extend(y2)
    Xtrain2 = np.array(Xtrain2)
    X2 = np.vstack([X2, Xtrain2])
    y2 = np.hstack([y2, ytrain2])
    X2 = X2[:X1.shape[0], :, :, :]
    y2 = y2[:X1.shape[0]]
    for i in range(X1.shape[0]):
        alpha = np.random.uniform(0.9, 1.1)
        noise = np.random.normal(loc=0., scale=1.0, size=X2[i, :, :, :].shape)
        X2[i, :, :, :] = alpha * X2[i, :, :, :] + 1 / 25 * noise
    if Xtrain1_super.shape[0] > Xtrain2_few.shape[0]:
        Xtrain1_super = copy.deepcopy(X1)
        Xtrain2_few = copy.deepcopy(X2)
        ytrain1_super = copy.deepcopy(y1)
        ytrain2_few = copy.deepcopy(y2)
    else:
        Xtrain2_few = copy.deepcopy(X1)
        Xtrain1_super = copy.deepcopy(X2)
        ytrain2_few = copy.deepcopy(y1)
        ytrain1_super = copy.deepcopy(y2)
    return Xtrain1_super, ytrain1_super, Xtrain2_few, ytrain2_few


def trainload(Xtrain1, ytrain1, Xtrain2, ytrain2, Xtrain1_super, ytrain1_super, Xtrain2_few, ytrain2_few, few_num,
              train_num):
    trainset1 = Train1(Xtrain1, ytrain1)
    train_loader1 = torch.utils.data.DataLoader(dataset=trainset1, batch_size=128, shuffle=True, num_workers=0)

    trainset2 = Train1(Xtrain2, ytrain2)
    train_loader2 = torch.utils.data.DataLoader(dataset=trainset2, batch_size=128, shuffle=True, num_workers=0)

    trainset3 = Train(Xtrain2_few, ytrain2_few)
    train_loader3 = torch.utils.data.DataLoader(dataset=trainset3, batch_size=64, shuffle=True, num_workers=0)

    trainset4 = Train(Xtrain1_super, ytrain1_super)
    train_loader4 = torch.utils.data.DataLoader(dataset=trainset4, batch_size=64, shuffle=True, num_workers=0)
    return train_loader1, train_loader2, train_loader3, train_loader4

X1, X2, y_label1, y_label2, root1 = dataload()

shapeor1 = X1.shape
data1 = X1.reshape(-1, X1.shape[-1])
data1 = StandardScaler().fit_transform(data1)
X1 = data1.reshape(shapeor1)
cla_flag1 = [np.sum(y_label1 == i) for i in range(1, np.max(y_label1) + 1)]
Location1, Labels1 = createImageCubes(X1, y_label1, cla_flag1, patch_size)
Xtrain1, ytrain1, _, _ = AugData_split(X1, cla_flag1, 200, Location1, Labels1, patch_size)
Xtrain1_super, ytrain1_super, test_Location1, test_Labels1 = AugData_split(X1, cla_flag1, train_num, Location1,
                                                                               Labels1, patch_size)

shapeor2 = X2.shape
data2 = X2.reshape(-1, X2.shape[-1])
data2 = StandardScaler().fit_transform(data2)
X2 = data2.reshape(shapeor2)
cla_flag2 = [np.sum(y_label2 == i) for i in range(1, np.max(y_label2) + 1)]
Location2, Labels2 = createImageCubes(X2, y_label2, cla_flag2, patch_size)
Xtrain2, ytrain2, _, _ = AugData_split(X2, cla_flag2, 200, Location2, Labels2, patch_size)
Xtrain2_few, ytrain2_few, test_Location2, test_Labels2 = AugData_split(X2, cla_flag2, few_num, Location2, Labels2,
                                                                           patch_size)

Xtrain1_super, ytrain1_super, Xtrain2_few, ytrain2_few = data_augment(Xtrain1_super, ytrain1_super, Xtrain2_few,
                                                                          ytrain2_few)

train_loader1, train_loader2, train_loader3, train_loader4 = trainload(Xtrain1, ytrain1, Xtrain2, ytrain2,
                                                                           Xtrain1_super, ytrain1_super, Xtrain2_few,
                                                                           ytrain2_few, few_num, train_num)


CLASS_NUM2 = np.max(y_label2)
criterion1 = nn.CrossEntropyLoss().to(device)
criterion2 = nn.MSELoss().to(device)
criterion3 = TripletLoss(margin=1).to(device)


class SELayer(nn.Module):
    def __init__(self, c1):
        super(SELayer, self).__init__()

        c1 = 32
        c = int(2 * (c1))

        self.cov1 = nn.Conv3d(c1, int(c1 / 2), kernel_size=1)
        self.cov2 = nn.Conv3d(c, int(c1 / 2), kernel_size=1)
        self.cov3 = nn.Conv3d(c1, c1, kernel_size=1)
        self.cov4 = nn.Conv3d(c1, c1, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x1_1 = self.cov1(x1)
        x2_1 = self.cov1(x2)
        x_1 = self.cov2(x)
        x1_2 = torch.cat((x_1, x1_1), 1)
        x2_2 = torch.cat((x_1, x2_1), 1)
        x1_3 = self.cov3(x1_2)
        x2_3 = self.cov3(x2_2)
        return x1_3, x2_3

class Classifier1(nn.Module):

    def __init__(self):
        super(Classifier1, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(100, 9),
            nn.Dropout(0.5),

        )

    def forward(self, x):
        c = self.fc1(x)
        return c


class Classifier2(nn.Module):

    def __init__(self):
        super(Classifier2, self).__init__()
        self.fc2 = nn.Sequential(
            nn.Linear(100, 9),
            nn.Dropout(0.5),

        )

    def forward(self, x):
        c = self.fc2(x)
        return c


class featureextraction1(nn.Module):
    def __init__(self, num_comp, c):
        super(featureextraction1, self).__init__()
        num = int(num_comp / 4 * c)
        self.layerc1 = nn.Sequential(
            nn.Linear(num, 100),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.layerc1(x)


class featureextraction2(nn.Module):
    def __init__(self, num_comp, c):
        super(featureextraction2, self).__init__()
        num = int(num_comp / 4 * c)
        self.layerd1 = nn.Sequential(
            nn.Linear(num, 100),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.layerd1(x)


class encoder(nn.Module):

    def __init__(self, window, num_comp):
        super(encoder, self).__init__()

        sconv = 32  # 共享特征卷积核数量
        dconv = 32  # 自编码器卷积核数量
        cconv = 32  # 分类器卷积核数量

        self.fea_extraction1 = featureextraction1(num_comp, sconv)
        self.fea_extraction2 = featureextraction2(num_comp, sconv)
        self.classifier1 = Classifier1()
        self.classifier2 = Classifier2()

        # 自编码器
        ''' d0 '''
        self.convd0 = nn.Sequential(
            nn.Conv3d(1, dconv, kernel_size=[window, window, 3], stride=1, padding=(0, 0, 1)),  # 16*22*22
            nn.BatchNorm3d(dconv),
            nn.LeakyReLU(),
        )
        self.se_layerd0 = SELayer(sconv)

        ''' d1 '''
        self.convd1 = nn.Sequential(
            nn.Conv3d(dconv + sconv, dconv, kernel_size=[1, 1, 3], stride=2, padding=(0, 0, 1)),  # 16*22*22
            nn.BatchNorm3d(dconv),
            nn.LeakyReLU(),
        )
        self.se_layerd1 = SELayer(sconv)

        ''' d2 '''
        self.convd2 = nn.Sequential(
            nn.Conv3d(dconv + sconv, dconv, kernel_size=[1, 1, 3], stride=2, padding=(0, 0, 1)),  # 4*9*9
            nn.BatchNorm3d(dconv),
            nn.LeakyReLU(),
        )
        self.se_layerd2 = SELayer(sconv)

        ''' d3 '''
        self.convd3 = nn.Sequential(
            nn.Conv3d(dconv + sconv, dconv, kernel_size=[1, 1, 3], stride=2, padding=(0, 0, 1)),  # 4*3*3
            nn.BatchNorm3d(dconv),
            nn.LeakyReLU(),
        )
        self.se_layerd3 = SELayer(sconv)

        # 分类网络
        ''' c0 '''
        self.convc0 = nn.Sequential(
            nn.Conv3d(1, cconv, kernel_size=[window, window, 3], stride=1, padding=(0, 0, 1)),  # 16*22*22
            nn.BatchNorm3d(cconv),
            nn.LeakyReLU(),
        )
        self.se_layerc0 = SELayer(sconv)

        ''' c1 '''
        self.convc1 = nn.Sequential(
            nn.Conv3d(cconv + sconv, cconv, kernel_size=[1, 1, 3], stride=2, padding=(0, 0, 1)),  # 16*22*22
            nn.BatchNorm3d(cconv),
            nn.LeakyReLU(),
        )
        self.se_layerc1 = SELayer(sconv)

        ''' c2 '''
        self.convc2 = nn.Sequential(
            nn.Conv3d(cconv + sconv, cconv, kernel_size=[1, 1, 3], stride=2, padding=(0, 0, 1)),  # 4*9*9
            nn.BatchNorm3d(cconv),
            nn.LeakyReLU(),
        )
        self.se_layerc2 = SELayer(sconv)

        ''' c3 '''
        self.convc3 = nn.Sequential(
            nn.Conv3d(cconv + sconv, cconv, kernel_size=[1, 1, 3], stride=2, padding=(0, 0, 1)),  # 4*3*3
            nn.BatchNorm3d(cconv),
            nn.LeakyReLU(),
        )
        self.se_layerc3 = SELayer(sconv)

        # 共享特征网络

        ''' s0 '''
        self.convs0 = nn.Sequential(
            nn.Conv3d(1, sconv, kernel_size=[window, window, 3], stride=1, padding=(0, 0, 1)),  # 16*22*22
            nn.BatchNorm3d(sconv),
            nn.LeakyReLU(),
        )

        ''' s1 '''
        self.convs1 = nn.Sequential(
            nn.Conv3d(cconv + sconv, sconv, kernel_size=[1, 1, 3], stride=2, padding=(0, 0, 1)),  # 16*22*22
            nn.BatchNorm3d(sconv),
            nn.LeakyReLU(),
        )
        # 自编码器网络
        ''' s2 '''
        self.convs2 = nn.Sequential(
            nn.Conv3d(cconv + sconv, sconv, kernel_size=[1, 1, 3], stride=2, padding=(0, 0, 1)),  # 4*9*9
            nn.BatchNorm3d(sconv),
            nn.LeakyReLU(),
        )

        ''' s3 '''
        self.convs3 = nn.Sequential(
            nn.Conv3d(cconv + sconv, sconv, kernel_size=[1, 1, 3], stride=2, padding=(0, 0, 1)),  # 4*3*3
            nn.BatchNorm3d(sconv),
            nn.LeakyReLU(),
        )

    def forward(self, x1, x2):
        outc0 = self.convc0(x1)
        outs0 = self.convs0(x1)
        attcs0, attsc0 = self.se_layerc0(outc0, outs0)
        aoutcs0 = outc0 * attcs0
        aoutsc0 = outs0 * attsc0
        inc1 = torch.cat((outc0, aoutsc0), 1)
        ins1 = torch.cat((outs0, aoutcs0), 1)

        outc1 = self.convc1(inc1)
        outs1 = self.convs1(ins1)
        attcs1, attsc1 = self.se_layerc1(outc1, outs1)
        aoutcs1 = outc1 * attcs1
        aoutsc1 = outs1 * attsc1
        inc2 = torch.cat((outc1, aoutsc1), 1)
        ins2 = torch.cat((outs1, aoutcs1), 1)

        outc2 = self.convc2(inc2)
        outs2 = self.convs2(ins2)
        attcs2, attsc2 = self.se_layerc2(outc2, outs2)
        aoutcs2 = outc2 * attcs2
        aoutsc2 = outs2 * attsc2
        inc3 = torch.cat((outc2, aoutsc2), 1)
        ins3 = torch.cat((outs2, aoutcs2), 1)

        outc3 = self.convc3(inc3)
        outs3 = self.convs3(ins3)
        attcs3, attsc3 = self.se_layerc3(outc3, outs3)
        aoutcs3 = outc3 * attcs3
        aoutsc3 = outs3 * attsc3
        inc4 = torch.cat((outc3, aoutsc3), 1)
        insc4 = torch.cat((outs3, aoutcs3), 1)

        outd0 = self.convd0(x2)
        outs0 = self.convs0(x2)

        attds0, attsd0 = self.se_layerd0(outd0, outs0)
        aoutds0 = outd0 * attds0
        aoutsd0 = outs0 * attsd0
        ind1 = torch.cat((outd0, aoutsd0), 1)
        ins1 = torch.cat((outs0, aoutds0), 1)

        outd1 = self.convd1(ind1)
        outs1 = self.convs1(ins1)
        attds1, attsd1 = self.se_layerd1(outd1, outs1)
        aoutds1 = outd1 * attds1
        aoutsd1 = outs1 * attsd1
        ind2 = torch.cat((outd1, aoutsd1), 1)
        ins2 = torch.cat((outs1, aoutds1), 1)

        outd2 = self.convd2(ind2)
        outs2 = self.convs2(ins2)
        attds2, attsd2 = self.se_layerd2(outd2, outs2)
        aoutds2 = outd2 * attds2
        aoutsd2 = outs2 * attsd2
        ind3 = torch.cat((outd2, aoutsd2), 1)
        ins3 = torch.cat((outs2, aoutds2), 1)

        outd3 = self.convd3(ind3)
        outs3 = self.convs3(ins3)
        attds3, attsd3 = self.se_layerd3(outd3, outs3)
        aoutds3 = outd3 * attds3
        aoutsd3 = outs3 * attsd3
        ind4 = torch.cat((outd3, aoutsd3), 1)
        insd4 = torch.cat((outs3, aoutds3), 1)

        f = inc4.view(inc4.size(0), -1)
        f = self.fea_extraction1(f)
        c = self.classifier1(f)

        f1 = ind4.view(ind4.size(0), -1)
        f1 = self.fea_extraction2(f1)
        c1 = self.classifier2(f1)

        return inc4, insc4, ind4, insd4, f, f1, c, c1


class decoder(nn.Module):

    def __init__(self, window, num_comp):
        super(decoder, self).__init__()

        sconv = 32  # 共享特征卷积核数量
        dconv = 32  # 自编码器卷积核数量
        cconv = 32  # 分类器卷积核数量

        num = window * window * num_comp

        self.fc1 = Classifier1()
        self.fc2 = Classifier2()
        # 自编码器
        ''' d4 '''
        self.convd4 = nn.Sequential(
            nn.ConvTranspose3d(dconv + sconv, dconv, kernel_size=[1, 1, 4], stride=2, padding=(0, 0, 1)),  # 1*9*9
            nn.BatchNorm3d(dconv),
            nn.LeakyReLU(inplace=True),
        )
        self.se_layerd4 = SELayer(sconv)

        ''' d5 '''
        self.convd5 = nn.Sequential(
            nn.ConvTranspose3d(dconv + sconv, dconv, kernel_size=[1, 1, 4], stride=2, padding=(0, 0, 1)),  # 1*9*9
            nn.BatchNorm3d(dconv),
            nn.LeakyReLU(),
        )
        self.se_layerd5 = SELayer(sconv)

        ''' d6 '''
        self.convd6 = nn.Sequential(
            nn.ConvTranspose3d(dconv, 1, kernel_size=[window, window, 4], stride=2, padding=(0, 0, 1)),  # 1*9*9
            nn.BatchNorm3d(1),
            nn.Tanh(),
        )

        # 分类网络
        ''' c4 '''
        self.convc4 = nn.Sequential(
            nn.ConvTranspose3d(cconv + sconv, cconv, kernel_size=[1, 1, 4], stride=2, padding=(0, 0, 1)),  # 1*9*9
            nn.BatchNorm3d(cconv),
            nn.LeakyReLU(),
        )
        self.se_layerc4 = SELayer(sconv)

        ''' c5 '''
        self.convc5 = nn.Sequential(
            nn.ConvTranspose3d(cconv + sconv, cconv, kernel_size=[1, 1, 4], stride=2, padding=(0, 0, 1)),  # 1*9*9
            nn.BatchNorm3d(cconv),
            nn.LeakyReLU(),
        )
        self.se_layerc5 = SELayer(sconv)

        ''' c6 '''
        self.convc6 = nn.Sequential(
            nn.ConvTranspose3d(cconv, 1, kernel_size=[window, window, 4], stride=2, padding=(0, 0, 1)),  # 1*9*9
            nn.BatchNorm3d(1),
            nn.Tanh(),
        )

        # 共享特征网络
        ''' s4 '''
        self.convs4 = nn.Sequential(
            nn.ConvTranspose3d(cconv + sconv, sconv, kernel_size=[1, 1, 4], stride=2, padding=(0, 0, 1)),  # 1*9*9
            nn.BatchNorm3d(sconv),
            nn.LeakyReLU(),
        )

        ''' s5 '''
        self.convs5 = nn.Sequential(
            nn.ConvTranspose3d((cconv + sconv), sconv, kernel_size=[1, 1, 4], stride=2, padding=(0, 0, 1)),  # 1*9*9
            nn.BatchNorm3d(sconv),
            nn.LeakyReLU(),
        )

        ''' s6 '''
        self.convs6 = nn.Sequential(
            nn.ConvTranspose3d(cconv, 1, kernel_size=[window, window, 4], stride=2, padding=(0, 0, 1)),  # 1*9*9
            nn.BatchNorm3d(1),
            nn.LeakyReLU(),
        )

        self.layer1 = nn.Sequential(
            nn.Linear(num, 100),
            nn.Dropout(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(100, 1),
            nn.Dropout(),
        )

    def forward(self, x1, x01, x2, x02):
        inc4 = x1
        ins4 = x01

        outc4 = self.convc4(inc4)
        outs4 = self.convs4(ins4)
        attcs4, attsc4 = self.se_layerc4(outc4, outs4)
        aoutcs4 = outc4 * attcs4
        aoutsc4 = outs4 * attsc4
        inc5 = torch.cat((outc4, aoutsc4), 1)
        ins5 = torch.cat((outs4, aoutcs4), 1)

        outc5 = self.convc5(inc5)
        outs5 = self.convs5(ins5)
        d = self.convc6(outc5)
        ds = self.convs6(outs5)
        ds = ds.view(ds.size(0), -1)
        ds = self.layer1(ds)

        ind4 = x2
        ins4 = x02

        outd4 = self.convd4(ind4)
        outs4 = self.convs4(ins4)
        attds4, attsd4 = self.se_layerd4(outd4, outs4)
        aoutds4 = outd4 * attds4
        aoutsd4 = outs4 * attsd4
        ind5 = torch.cat((outd4, aoutsd4), 1)
        ins5 = torch.cat((outs4, aoutds4), 1)

        outd5 = self.convd5(ind5)
        outs5 = self.convs5(ins5)
        d1 = self.convd6(outd5)
        ds1 = self.convs6(outs5)
        ds1 = ds1.view(ds1.size(0), -1)
        ds1 = self.layer1(ds1)

        return d, d1, ds, ds1

encoder = encoder(patch_size, numComponents).to(device)
decoder = decoder(patch_size, numComponents).to(device)


decoder_optimizier = torch.optim.Adam(decoder.parameters(), lr=0.001, eps=1e-3)
encoder_optimizier = torch.optim.Adam(encoder.parameters(), lr=0.001, eps=1e-3)
decoder_scheduler \
    = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizier, mode='min', factor=0.5, patience=5, threshold=1e-4,
                                                 threshold_mode='abs', cooldown=0, min_lr=0.001, eps=1e-8)
encoder_scheduler \
    = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizier, mode='min', factor=0.5, patience=5, threshold=1e-4,
                                                 threshold_mode='abs', cooldown=0, min_lr=0.001, eps=1e-8)

for it in range(epoches):
    for indx, data in enumerate(zip(train_loader1, train_loader2, cycle(train_loader4), cycle(train_loader3))):
        image1 = data[0][0].to(device)  # image为PU
        label1 = data[0][1].to(device)

        image2 = data[1][0].to(device)  # image为PU
        label2 = data[1][1].to(device)

        image3 = data[2][0].to(device)  # image为PU
        label3 = data[2][1].to(device)

        image4 = data[3][0].to(device)  # image为PU
        label4 = data[3][1].to(device)

        fea1, feas1, fea2, feas2, _, _, _, _ = encoder(image1, image2)
        d1, d2, ds1, ds2 = decoder(fea1, feas1, fea2, feas2)
        loss_recon = criterion2(d1, label1) + criterion2(d2, label2)

        fea3, feas3, fea4, feas4, f3, f4, c3, c4 = encoder(image3, image4)
        d3, d4, ds3, ds4 = decoder(fea3, feas3, fea4, feas4)
        f_3 = torch.cat((f3, ds3), 1)
        f_4 = torch.cat((f4, ds4), 1)

        loss_dis = criterion3(f_3, label3) + criterion3(f_4, label4)
        loss_class = criterion1(c3, label3) + criterion1(c4, label4)
        loss = loss_dis + loss_recon + loss_class
        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()
        encoder_optimizier.step()
        decoder_optimizier.step()


    encoder_scheduler.step(it)
    decoder_scheduler.step(it)

    print('第{}轮的loss = {}, loss_recon = {}， loss_dis = {}, loss_class = {}'.format(
        it + 1, loss.cpu().item(), loss_recon.cpu().item(), loss_class.cpu().item(), loss_dis.cpu().item()))

print('Testing Start')
ref_samples1, ref_samples2, ref_labels1, ref_labels2 = anchor_feature(train_loader3, train_loader4, device, encoder,
                                                                      decoder, encoder_optimizier, decoder_optimizier)
class_num1 = np.max(ref_labels1) + 1
class_num2 = np.max(ref_labels2) + 1
ref_samples1 = ref_samples1[:train_num * class_num1]
ref_labels1 = ref_labels1[:train_num * class_num1]
ref_samples2 = ref_samples2[:few_num * class_num2]
ref_labels2 = ref_labels2[:few_num * class_num2]
ans1, ans2, fea1, fea2 = test(X1, X2, patch_size, device, encoder, decoder, encoder_optimizier, decoder_optimizier)

label_x1, label_x2, OA1, OA2, kappa1, kappa2, classification1, classification2 \
    = accuracycompute(ref_samples1, ref_samples2, ref_labels1, ref_labels2, ans1, ans2, fea1, fea2, y_label1, y_label2, Labels1,Labels2)

print('Supervised Task')
print('OA:', OA1)
print('Kappa:', kappa1)
print('AA:', classification1)  # 在classification_report中，recall表示每一类的准确度，accuracy为OA，recall对应的macro avg表示AA
print('Few-shot Task')
print('OA:', OA2)
print('Kappa:', kappa2)
print('AA:', classification2)  # 在classification_report中，recall表示每一类的准确度，accuracy为OA，recall对应的macro avg表示AA


gt1 = sio.loadmat(root + '//denoiseHSI//PaviaU_gt.mat')['paviaU_gt']
gt_color1 = [[0, 0, 0], [252, 226, 223], [81, 184, 70], [136, 204, 234],
             [38, 139, 67], [159, 93, 166], [161, 82, 44], [128, 70, 156],
             [238, 33, 35], [246, 236, 20]]
r1, c1 = np.shape(gt1)
list_result1 = np.empty((r1, c1, 3))
result1 = np.empty((r1, c1))
m1 = 0
for i in range(r1):
    for j in range(c1):
        list_result1[i][j][0] = gt_color1[int(label_x1[m1] + 1)][0]
        list_result1[i][j][1] = gt_color1[int(label_x1[m1] + 1)][1]
        list_result1[i][j][2] = gt_color1[int(label_x1[m1] + 1)][2]
        result1[i][j] = label_x1[m1] + 1
        m1 = m1 + 1
list_result1 = list_result1.astype(np.uint8)
result1 = result1.astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow(list_result1)

sio.savemat(root1 + '//mtlpaviau.mat', mdict={'my_data': list_result1})
sio.savemat(root1 + '//mtlpaviau_category.mat', mdict={'my_data': result1})

gt2 = sio.loadmat(root + '//denoiseHSI//Paviac_gt.mat')['pavia_gt']
gt_color2 = [[0, 0, 0], [179, 179, 180], [0, 255, 0], [102, 255, 255],
                 [0, 129, 0], [255, 48, 205], [154, 102, 48],
                 [154, 0, 154], [255, 0, 0], [255, 255, 0]]

r2, c2 = np.shape(gt2)
list_result2 = np.empty((r2, c2, 3))
result2 = np.empty((r2, c2))
m2 = 0
for i in range(r2):
    for j in range(c2):
        list_result2[i][j][0] = gt_color2[int(label_x2[m2] + 1)][0]
        list_result2[i][j][1] = gt_color2[int(label_x2[m2] + 1)][1]
        list_result2[i][j][2] = gt_color2[int(label_x2[m2] + 1)][2]
        result2[i][j] = label_x2[m2] + 1
        m2 = m2 + 1
list_result2 = list_result2.astype(np.uint8)
result2 = result2.astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow(list_result2)

sio.savemat(root1 + '//mtlpaviac.mat', mdict={'my_data': list_result2})
sio.savemat(root1 + '//mtlpaviac_category.mat', mdict={'my_data': result2})
