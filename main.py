#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/11 下午4:28
# @Author  : PeiP Liu
# @FileName: main.py
# @Software: PyCharm

import os
import torch
import pickle
import torch.nn as nn
import numpy as np
from tqdm import trange
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from arguments import BilstmCnnArgs as args
from arguments import BertArgs as bert_args
from BiLSTM_CNN.model import build_model
from data_utils import gen_batch_data
from Finetuning_BertCRF.BertModel import BERT_CRF_NER

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_sents = (args.train_wordids_pad, args.train_charids_pad, args.train_posids_pad, args.train_caseids_pad)
    train_labels = args.train_labelids_pad

    valid_sents = (args.valid_wordids_pad, args.valid_charids_pad, args.valid_posids_pad, args.valid_caseids_pad)
    valid_labels = args.valid_labelids_pad

    test_sents = (args.test_wordids_pad, args.test_charids_pad, args.test_posids_pad, args.test_caseids_pad)
    test_labels = args.test_labelids_pad

    bert_train_sents = bert_args.train_seq_list
    bert_train_labels = bert_args.train_seq_label_list

    bert_valid_sents = bert_args.valid_seq_list
    bert_valid_labels = bert_args.valid_seq_label_list

    bert_test_sents = bert_args.test_seq_list
    bert_test_labels = bert_args.test_seq_label_list

    word2indx = args.word2idx
    label2idx = args.label2idx

    writer = SummaryWriter(log_dir='../Result/LSTM_model', comment='scalar_record')
    early_stop = EarlyStopping(monitor='acc', min_delta=args.min_delta, patience=args.patience)

    model = build_model('multi_feature_bilstm_atten_crf', args).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_decay_factor,
                                                    verbose=True, patience=5, min_lr=args.min_lr)  # 重点关注

    """
    if all(map(os.path.exists, 'Result/Embedding/word_embedding.npy')):
        pretrained_embedding = np.load('Result/Embedding/word_embedding.npy')
        model.init_embedding(pretrained_embedding)
    """

    for epoch in trange(args.total_train_epoch, desc='Epoch'):
        model.train()
        for i_batch, (batch_train_sents, batch_train_labels, batch_bert_train_sents, batch_bert_train_labels) in \
                enumerate(gen_batch_data(train_sents, train_labels, bert_train_sents, bert_train_labels, args.num_train, args.batch_size)):
            i_batch_train_sents = torch.from_numpy(batch_train_sents).long().to(args.device)
            i_batch_train_labels = torch.from_numpy(batch_train_labels).long().to(args.device)
            i_batch_bert_train_sents = batch_bert_train_sents.tolist()
            i_batch_bert_train_labels = batch_bert_train_labels.tolist()
