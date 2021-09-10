#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/11 上午11:18
# @Author  : PeiP Liu
# @FileName: model_evaluation.py
# @Software: PyCharm
import time
import datetime
import numpy as np
import torch


def time_format(time_diff):
    seconds = int(round(time_diff))
    return str(datetime.timedelta(seconds=seconds))


def self_f1_score(y_pred, y_true):
    # 0,1,2,3 are [CLS],[SEP],[X],[O]
    ignore_id = 3
    # get the num of valid prediction in y_pred
    num_proposed = len(y_pred[y_pred > ignore_id])
    # the num of correct prediction
    num_correct = (np.logical_and(y_true == y_pred, y_true > ignore_id)).sum()
    # the num of valid groudtruth
    num_gold = len(y_true[y_true > ignore_id])
    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0
    try:
        f1 = 2 * recall * precision / (recall + precision)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0
    return precision, recall, f1


def bert_evaluate(eva_model, eva_dataloader, eva_epoch_th, eva_device, eva_dataset_name):
    eva_model.eval()
    all_pred_labels = []
    all_true_labels = []
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for eva_batch in eva_dataloader:
            # we move the data to specific device
            eva_batch_data = tuple(item.to(eva_device) for item in eva_batch)
            # each batch_data contains several kinds of infor
            eva_input_ids, eva_true_mask, eva_seg_ids, eva_pre_mask, eva_true_label_ids, eva_true_label_mask = eva_batch_data
            # input to model to get the predicted result
            _, pred_labels_ids = eva_model(eva_input_ids, eva_true_mask, eva_seg_ids, eva_pre_mask,
                                           eva_true_label_mask)
            # getting the valid result based on mask
            valid_preds = torch.masked_select(pred_labels_ids.detach().cpu(), eva_true_label_mask.cpu().bool())
            valid_true = torch.masked_select(eva_true_label_ids.to('cpu'), eva_true_label_mask.cpu().bool())
            assert len(valid_preds) == len(valid_true)
            # convert the data type to list, and fuse them to a unified global list
            all_pred_labels.extend(valid_preds.tolist())
            all_true_labels.extend(valid_true.tolist())
            # the all num of valid infor
            total = total + len(valid_true)
            # all num of equal(all_pred_labels, all_true_labels)
            correct = correct + valid_preds.eq(valid_true).sum().item()
            # correct += (valid_preds==valid_true).sum().item()
    average_acc = correct / total
    assert len(all_true_labels) == len(all_pred_labels)
    precision, recall, f1 = self_f1_score(np.array(all_pred_labels), np.array(all_true_labels))
    # we also can compute the score by using packages from sklearn.metrics, such as f1_score
    # f1 = f1_score(y_true=np.array(all_true_labels), y_pred=np.array(all_pred_labels))
    end = time.time()
    print("This is %s:\n Epoch:%d\n Acc:%.2f\n Precision: %.2f\n Recall:%.2f\n F1: %.2f\n Spending: %s" % \
          (eva_dataset_name, eva_epoch_th, average_acc * 100., precision * 100., recall * 100., f1 * 100.,
           time_format(end - start)))
    return average_acc, f1


# def multi_scale_evaluate():