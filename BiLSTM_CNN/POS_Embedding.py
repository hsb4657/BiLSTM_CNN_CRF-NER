#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/31 下午2:41
# @Author  : PeiP Liu
# @FileName: POS_Embedding.py
# @Software: PyCharm

import torch
import os
import json
import pickle
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("..")
from arguments import BilstmCnnArgs as args

embedding_dim = args.pos_embed_dim
windows_size = 5
sentences = args.all_posids_pad  # all the posids_sentences with padding after text2ids
vocab_size = len(args.pos2idx)  # the size of pos2index


class PosEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        x = self.embedding(x)  # 相当于lookup
        x = self.fc(x)  # 进行线性分类
        return F.log_softmax(x, dim=1)  # 相当于


def build_skip_grams(corpus, windows_size):
    skip_grams = []

    for tokens in corpus:  # 表示一个句子
        for i, center in enumerate(tokens):
            left = max(0, i - windows_size)
            right = min(i+windows_size, len(tokens))
            for j in range(left, right):
                if j != i:
                    skip_grams.append((center, tokens[j]))
    return skip_grams


skip_grams = build_skip_grams(sentences, windows_size)


def gen_batch(data, batch_size):
    data = np.array(data)
    data_len = len(data)
    indices = np.arange(data_len)
    np.random.shuffle(indices)

    indx = 0

    while True:
        if indx + batch_size >= data_len:
            batch = data[indices[indx:]]
            center_batch = batch[:, 0]
            context_batch = batch[:, 1]

            yield center_batch, context_batch
            break
        else:
            batch = data[indices[indx: indx + batch_size]]
            center_batch = batch[:, 0]
            context_batch = batch[:, 1]

            yield center_batch, context_batch
            indx = indx + batch_size


if __name__ == '__main__':
    if not os.path.exists('..Result/Embedding'):
        os.makedirs('..Result/Embedding')

    lr = 0.001
    batch_size = 64
    num_epoch = 30

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PosEmbedding(vocab_size, embedding_dim).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    print("Starting the pos_emb training...")
    for epoch in range(1, num_epoch+1):
        for i, (centers, contexts) in enumerate(gen_batch(skip_grams, batch_size)):
            # 软件、组件、第三方库、漏洞；当前采用的方法，存在ＸＸＸ的问题，我们采用ＸＸＸ方法，可以实现ＸＸＸ效果
            centers = torch.from_numpy(centers).long().to(device)
            contexts = torch.from_numpy(contexts).long().to(device)

            out = model(centers)  # 模型主要是为了获取邻居信息，即上下文
            loss = criterion(out, contexts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))
        np.save(os.path.join('..Result/Embedding', 'pos_{:0>2d}.npy'.format(epoch)), model.embedding.weight.detach().cpu().numpy())
        if os.path.exists(os.path.join('..Result/Embedding', 'pos_{:0>2d}.npy'.format(epoch-3))):
            os.remove(os.path.join('..Result/Embedding', 'pos_{:0>2d}.npy'.format(epoch-3)))

    # 文件中存储的向量即是pos_emb，第一行向量即对应idx2pos[0]
    np.save('..Result/Embedding/pos_embedding.npy', model.embedding.weight.detach().cpu().numpy())
    print('POS training is Over !')
