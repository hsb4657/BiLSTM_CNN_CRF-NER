#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 上午10:00
# @Author  : PeiP Liu
# @FileName: Bert_Feature.py
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from .Bert_data_utils import DataProcessor, BertCRFData
from .BertModel import BERT_CRF_NER
import sys
sys.path.append("..")
from arguments import BertArgs as s_args
from arguments import BilstmCnnArgs as bc_args


class GetBertFeature:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(s_args.model_list[s_args.model_id], do_lower_case=False)

        bert_model = BertModel.from_pretrained(s_args.model_list[s_args.model_id])
        self.model = BERT_CRF_NER(bert_model, s_args.label2idx, batch_size=s_args.batch_size, max_seq_len=s_args.max_seq_len,
                             device=bc_args.device)
        checkpoint = torch.load(s_args.output_dir + 'bert_crf_ner.checkpoint.pt', map_location=bc_args.device)
        # parser the model params
        pretrained_model_dict = checkpoint['model_state']
        # get the model param names
        model_state_dict = self.model.state_dict()
        # get the params interacting between model_state_dict and pretrained_model_dict
        selected_model_state = {k: v for k, v in pretrained_model_dict.items() if k in model_state_dict}
        model_state_dict.update(selected_model_state)
        # load the params into model
        self.model.load_state_dict(model_state_dict)
        self.model.to(bc_args.device)

    def get_bert_feature(self, batch_sents, batch_labels, device):
        batch_dp = DataProcessor(batch_sents, batch_labels, self.tokenizer, s_args.max_seq_len, s_args.label2idx)
        batch_bert_data = BertCRFData(batch_dp.get_features())
        batch_dataloader = DataLoader(dataset=batch_bert_data, batch_size=s_args.batch_size, shuffle=False,
                                      collate_fn=BertCRFData.seq_tensor)  # return the iterator object of batch_data
        self.model.eval()
        with torch.no_grad():
            for i_batch_data in batch_dataloader:  # in fact, there is only one batch_data
                batch_data = tuple(t.to(device) for t in i_batch_data)
                input_ids, input_mask, seg_ids, pre_mask, label_ids, label_mask = batch_data
                bert_emission = self.model.get_bert_emission(input_ids, input_mask, seg_ids, pre_mask)
                return bert_emission


