#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/10 下午5:02
# @Author  : PeiP Liu
# @FileName: utils.py
# @Software: PyCharm


def decode_tag(emission, valid_mask):
    """
    :param emission: (batch_size, sent_len, num_labels)
    :param valid_mask: (batch_size, sent_len)
    :return:
    """
    valid_sentlen =valid_mask.sum(1)  # (batch_size,), the valid length of each sent in the batch-data
    pre_tag = emission.argmax(-1)  # (batch_size, sent_len, ), get the place of max ele in feature-dim
    pre_valid_tag = [pre_tag[i_sent][:valid_sentlen[i_sent].item()].detach().tolist() for i_sent in range(emission.size(0))]

    return pre_valid_tag
