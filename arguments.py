#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 下午3:43
# @Author  : PeiP Liu
# @FileName: arguments.py
# @Software: PyCharm
import torch


class BasicArgs:
    batch_size = 64
    max_seq_len = 512
    learning_rate = 1e-3
    # choose the device, if GPU is available, we can use it. otherwise, the cpu is a replacement
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    total_train_epoch = 10
    train_seq_list, train_seq_label_list
    valid_seq_list, valid_seq_label_list
    test_seq_list, test_seq_label_list
    label2idx
    idx2label
    word2idx
    idx2word
    pos2idx
    idx2pos


class BertArgs(BasicArgs):
    output_dir = 'Result/BERT'

    model_list = ['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased']
    model_id = 2

    # if you retrain the model, please make load_checkpoint = True
    load_checkpoint = False
    weight_decay_finetune = 1e-5
    lr_crf_fc = 1e-5
    weight_decay_crf_fc = 1e-5
    warmup_proportion = 0.002

    # the larger batch_size is, the effect will be better. However, some labs's GPU is not available enough.
    # So we can accumulate X*gradient (gradient in each batch) to achieve the same effect as batch_size*X.
    # that is, we don't empty the gradient each batch until X batches
    # rf https://blog.csdn.net/Princeicon/article/details/108058822
    gradient_accumulation_steps = 4


class BilstmCnnArgs(BasicArgs):
    char_embed_dim = 30
    word_embed_dim = 100
    pos_embed_dim = 20
    case_embed_dim = 7
    input_dim = 157  # char_emb_dim + word_emb_dim + pos_emb_dim + case_emb_dim
    hid_dim = 512
    model_dim = 256

    word_emb_table_file = ''  # 文件地址
    char_emb_table_file = ''  # 文件地址
    pos_emb_table_file = ''  # 文件地址
    case_emb_table_file = ''  # 文件地址

    word_pad_indx = 0
    label_pad_indx = 0
    label_bos_indx = 1
    label_eos_indx = 2

    transformer_num_blocks = 4
    transformer_num_heads = 8
    num_labels = len(label2inx)

    bilstm_layers = 4

    dropout_rate = 0.2
    attention_type = 'general'


