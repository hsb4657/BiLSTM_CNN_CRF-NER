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
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from arguments import BilstmCnnArgs as args
from arguments import BertArgs as bert_args
from BiLSTM_CNN.model import build_model
from data_utils import gen_batch_data
from Finetuning_BertCRF.Bert_Feature import GetBertFeature
from .common_modules.model_evaluation import lc_cal_f1, lc_cal_acc
from .common_modules.utils import EarlyStopping
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

    writer = SummaryWriter(log_dir=args.output_dir, comment='scalar_record')
    early_stop = EarlyStopping(monitor='acc', min_delta=args.min_delta, patience=args.patience)

    model = build_model('multi_feature_bilstm_atten_crf', args).to(args.device)
    bert_emission = GetBertFeature()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # rf https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_decay_factor,
                                                    verbose=True, patience=5, min_lr=args.min_lr)  # 重点关注

    """
    if all(map(os.path.exists, 'Result/Embedding/word_embedding.npy')):
        pretrained_embedding = np.load('Result/Embedding/word_embedding.npy')
        model.init_embedding(pretrained_embedding)
    """
    print("*****************************Starting Training*****************************")
    num_batch = args.num_train // args.batch_size if args.num_train % args.batch_size==0 else args.num_train//args.batch_size+1

    # record the training infor
    train_ave_loss = []
    valid_acc_score = []
    valid_f1_score = []
    valid_loss_score = []

    for epoch in trange(args.total_train_epoch, desc='Epoch'):
        train_loss = 0

        # compute the training time, and initiate the time
        train_start = time.time()
        batch_start = time.time()

        # setting the training mode and clear the grad
        model.train()
        model.zero_grad()
        for i_batch, (batch_train_sents, batch_train_labels, batch_bert_train_sents, batch_bert_train_labels) in \
                enumerate(gen_batch_data(train_sents, train_labels, bert_train_sents, bert_train_labels, args.num_train, args.batch_size)):

            # remove the data to device(GPU)
            i_batch_train_word_sentences = torch.from_numpy(batch_train_sents[0]).long().to(args.device)
            i_batch_train_char_sentences = torch.from_numpy(batch_train_sents[1]).long().to(args.device)
            i_batch_train_pos_sentencens = torch.from_numpy(batch_train_sents[2]).long().to(args.device)
            i_batch_train_case_sentences = torch.from_numpy(batch_train_sents[3]).long().to(args.device)
            i_batch_train_labels = torch.from_numpy(batch_train_labels).long().to(args.device)

            # bert feature
            i_batch_bert_train_sents = batch_bert_train_sents.tolist()
            i_batch_bert_train_labels = batch_bert_train_labels.tolist()
            batch_bert_feature = bert_emission.get_bert_feature(i_batch_bert_train_sents, i_batch_bert_train_labels, args.device)

            # input and output
            loss = model(i_batch_train_char_sentences, i_batch_train_word_sentences, i_batch_train_pos_sentencens,
                         i_batch_train_case_sentences, i_batch_train_labels, 'train', batch_bert_feature)

            # backpropagation and clear the grad
            loss.backward()
            train_loss = train_loss + loss.cpu().item()
            optimizer.step()
            optimizer.zero_grad()

            # compute the training time
            if i_batch % 10 == 0 and i_batch != 0:
                print('Ten batches cost time : {}'.format(time.time()-batch_start))
                # print the training infor
                print("Epoch:{}-{}/{}, Loss:{}".format(epoch, i_batch, num_batch, loss))
                batch_start = time.time()
                writer.add_scalar("train_loss", loss.cpu().item(), epoch*num_batch+i_batch)

        ave_loss = train_loss/num_batch  # the average loss of each epoch
        train_ave_loss.append(ave_loss)
        print("Epoch: {} is completed, the average loss is: {}, spend: {}".format(epoch,ave_loss,time.time()-train_start))
        print("********************Let us begin the validation of epoch {}***************************".format(epoch))

        # we save the model
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'ckpt_epoch_{:2d}.pt'.format(epoch)))
        if os.path.exists(os.path.join(args.output_dir, 'ckpt_epoch_{:2d}.pt'.format(epoch-args.patience-1))):
            os.remove(os.path.join(args.output_dir, 'ckpt_epoch_{:2d}.pt'.format(epoch-args.patience-1)))

        # evaluate the model
        model.eval()
        valid_true, valid_pre = [], []
        valid_acml_loss = 0
        for j_batch, (batch_valid_sents, batch_valid_labels, batch_bert_valid_sents, batch_bert_valid_labels) in \
                enumerate(gen_batch_data(valid_sents, valid_labels, bert_valid_sents, bert_valid_labels, args.num_valid, args.batch_size*2)):

            # remove the data to device(GPU)
            j_batch_valid_word_sentences = torch.from_numpy(batch_valid_sents[0]).long().to(args.device)
            j_batch_valid_char_sentences = torch.from_numpy(batch_valid_sents[1]).long().to(args.device)
            j_batch_valid_pos_sentences = torch.from_numpy(batch_valid_sents[2]).long().to(args.device)
            j_batch_valid_case_sentences = torch.from_numpy(batch_valid_sents[3]).long().to(args.device)
            j_batch_valid_labels = torch.from_numpy(batch_valid_labels).long().to(args.device)

            # bert feature
            j_batch_bert_valid_sents = batch_bert_valid_sents.tolist()
            j_batch_bert_valid_labels = batch_bert_valid_labels.tolist()
            batch_bert_feature = bert_emission.get_bert_feature(j_batch_bert_valid_sents, j_batch_bert_valid_labels, args.device)

            # input and output
            loss = model(j_batch_valid_char_sentences, j_batch_valid_word_sentences, j_batch_valid_pos_sentences,
                         j_batch_valid_case_sentences, j_batch_valid_labels, 'train', batch_bert_feature)

            # input and output
            _, preds = model(j_batch_valid_char_sentences, j_batch_valid_word_sentences, j_batch_valid_pos_sentences,
                         j_batch_valid_case_sentences, j_batch_valid_labels, 'test', batch_bert_feature)

            valid_true.extend([each_label for each_sent in batch_valid_labels for each_label in each_sent if each_label!=args.label_pad_indx])  # array is also well
            valid_pre.extend([each_pred_label for each_pre_sent in preds for each_pred_label in each_pre_sent])

            valid_acml_loss = valid_acml_loss + loss.detach().cpu().item()*len(j_batch_bert_valid_sents)

        valid_avg_loss = valid_acml_loss/args.num_valid
        valid_loss_score.append(valid_avg_loss)
        valid_f1_score.append(lc_cal_f1(valid_true, valid_pre))
        valid_acc_score.append(lc_cal_acc(true_tags=valid_true, pred_tags=valid_pre))
        print('Validation: Epoch-{}, Val_loss-{}, Val_acc-{}, Val_f1-{}'.format(epoch, valid_avg_loss, valid_acc_score[-1], valid_f1_score[-1]))

        writer.add_scalar('val-loss', valid_avg_loss, epoch)
        writer.add_scalar('val-f1', valid_f1_score[-1], epoch)
        writer.add_scalar('val-acc', valid_acc_score[-1], epoch)

        lr_decay.step(valid_avg_loss)  # when there is no change about loss within patience step , lr will decay

        if early_stop.judge(epoch, valid_f1_score[-1]):
            print("Early stop at epoch {}, with val-f1 score {}".format(epoch, valid_f1_score[-1]))
            print('Best performance epoch {}, with best val-f1 score {}'.format(early_stop.best_epoch, early_stop.best_val))
            break

    print("**********************************************\n"
          "********     The training is over.    ********\n"
          "**********************************************")

    # then, we will show the training and validation processing by figure.
    # set the plot style from seaborn
    sns.set(style='darkgrid')
    # increase the plot size(line width) and figure size
    sns.set(font_scale=1.5)
    plt.rcParams['figure.figsize'] = [12, 6]

    # plot the learning curve. the params are :values, color, line-title
    plt.plot(train_ave_loss, 'b-o', 'train_average_loss')  # epoch as the period
    plt.plot(valid_loss_score, 'm-o', 'valid_average_loss')
    plt.plot(valid_acc_score, 'r-o', 'valid_acc_score')
    plt.plot(valid_f1_score, 'g-o', 'valid_f1_score')

    # now we label the plot
    plt.title('Learning curve')
    plt.xlabel('Epoch')
    plt.ylabel('TrainLoss/ValLoss/ValAcc/ValF1')
    # legend() shows the line-title
    plt.legend()
    plt.show()
