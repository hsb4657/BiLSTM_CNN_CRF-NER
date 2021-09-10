#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 上午10:52
# @Author  : PeiP Liu
# @FileName: main.py
# @Software: PyCharm

import torch
import torch.nn as nn
from data_utils import *

# get the text data from orig_file
train_dataset = processing_orgdata('train')
dev_dataset = processing_orgdata('dev')
test_dataset = processing_orgdata('test')
# get the max length of sentence(word) and word(char)
sent_maxlen = max(train_dataset[4], dev_dataset[4], test_dataset[4]) # 这里的最长后续要根据数据的分布情况调整
word_maxlen = max(train_dataset[5], dev_dataset[5], test_dataset[5]) # 这里的最长后续要根据数据的分布情况调整

# construct the dict of
build_vocab_sentences = train_dataset[0] + dev_dataset[0] + test_dataset[0]
build_vocab_sentences_labels = train_dataset[1] + dev_dataset[1] + test_dataset[1]
build_vocab_sentences_pos = train_dataset[2] + dev_dataset[2] + test_dataset[2]
# the dict result，后续我们需要增强字符字典的内容
build_vocab_result = build_vocab(build_vocab_sentences, build_vocab_sentences_labels, build_vocab_sentences_pos)

case2idx, case_emb = case_feature()
# convert the orig_text to id
train_text2ids = text2ids(train_dataset[0], train_dataset[1], train_dataset[2], build_vocab_result[0],
                          build_vocab_result[2], build_vocab_result[4], build_vocab_result[6], case2idx)

# pad the word_id_sentence
train_word_sentence_padding = sentence_padding(train_text2ids[0], sent_maxlen, build_vocab_result[0]['[PAD]'])
train_word_sentence_padding = torch.tensor(train_word_sentence_padding, dtype=torch.long)
# pad the char_id_sentence
train_char_sentences_padding = char_sentences_padding(train_text2ids[1], sent_maxlen, word_maxlen)
train_char_sentences_padding = torch.tensor(train_char_sentences_padding, dtype=torch.long)


# get all the feature_tables
case_emb_table = torch.tensor(case_emb, dtype=torch.float32)
pos_emb_table = torch.tensor(build_pos_emb_table, dtype=torch.float32)
char_emb_table = torch.tensor(build_char_emb_table(build_vocab_result[3]), dtype=torch.float32)

glove = GloveFeature('/home/liupei/2021Paper/Dataset/glove.840B.300d.txt') # 该地址后续可能会变
glove_embedding_dict = glove.load_glove_embedding()
word_emb_table = build_word_emb_table(build_vocab_result[1], glove_embedding_dict, glove.glove_dim)
word_emb_table = torch.tensor(word_emb_table, dtype=torch.float32)
