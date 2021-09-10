import torch
from torch.utils.data import Dataset


class BertFormatData(Dataset):
    def __init__(self, mode, tokenizer):
        # make sure the data be for train or test
        assert mode in ['Train', 'Test']
        self.mode = mode
        self.data = readlines(file_path)
        self.len = len(self.data)
        self.label_map = label2index_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        if self.mode == 'Test':
            text_data = self.data
            label_tensor = None
        else:
            text_data = self.data[0]
            label = self.data[1]
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)

        word_pieces = ['[CLS]']
        token_data = self.tokenizer.tokenize(text_data)
        word_pieces = word_pieces + token_data +['[SEP]']
        token_length = len(word_pieces)

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        token_tensor = torch.tensor(ids)

        seg_tensor = torch.tensor([1]*token_length, dtype = torch.long)

        return token_tensor, seg_tensor, label_tensor

    def __len__(self):
        return self.len


train_dataset = BertFormatData('Train',tokenizer = tokenizer)