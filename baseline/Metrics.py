# coding=utf-8

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def score(pred,labels):
    pre = precision_score(labels, pred, average='macro')
    recall = recall_score(labels, pred, average='macro')
    f1 = f1_score(labels, pred, average='macro')
    return pre, recall, f1

def metric_acc(y, pred_y):
    num = len(y)
    acc = 0
    for i in range(num):
        if y[i] == pred_y[i]:
            acc += 1
    acc = float(acc) / num
    return acc
