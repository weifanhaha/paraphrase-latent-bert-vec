#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import re
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn


# In[2]:


from tqdm import tqdm
import re
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import os

class QuoraBertMaskPredictProbDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, 
                 text_path='../data/quora_train.txt', pretrained_model_name='bert-base-cased', preprocessed_folder=None):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        
        self.tokenizer = self.init_tokenizer(pretrained_model_name)
        self.mask_predict_model = BertForMaskedLM.from_pretrained(pretrained_model_name)
        self.sentences = self.read_text(text_path)
        self.init_constants()
        
        self.n_words = len(self.tokenizer)
        
        if preprocessed_folder is not None and os.path.exists(preprocessed_folder):
            self.preprocessed_folder = preprocessed_folder
        else:
            self.preprocessed_folder = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mask_predict_model = self.mask_predict_model.to(self.device)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        seq1, seq2 = sentence.split('\t')
        
        tokens1 = self.tokenizer.tokenize(seq1)
        word_pieces1 =  [self.SOS_token] + tokens1 + [self.EOS_token]
        idxes1 = self.tokenizer.convert_tokens_to_ids(word_pieces1)
        
        tokens2 = self.tokenizer.tokenize(seq2)
        word_pieces2 = [self.SOS_token] + tokens2 + [self.EOS_token]
        idxes2 = self.tokenizer.convert_tokens_to_ids(word_pieces2)
        
        seq1_tensor = torch.tensor(idxes1, dtype=torch.long)
        seq2_tensor = torch.tensor(idxes2, dtype=torch.long)
        
        if self.preprocessed_folder is not None:
            if self.mode == 'train':
                pretrained_idx = idx
            elif self.mode == 'val':
                pretrained_idx = self.train_size + idx
            else:
                pretrained_idx = self.train_size + self.val_size + idx
            preprocessed_file = "{}/{}.npy".format(self.preprocessed_folder, pretrained_idx)
            mask_probs = np.load(preprocessed_file, allow_pickle=True)
            mask_probs = torch.tensor(mask_probs)
        else:
            # mask each word from the begining to the end
            # get the probability distribution of the mask tokens
            pred = self.get_mask_pred_probs(seq1_tensor)
            mask_probs = self.get_stack_probs(pred)
        
        return seq1_tensor, seq2_tensor, mask_probs

    def __len__(self):
        if self.mode == 'train':
            return self.train_size
        elif self.mode == 'val':
            return self.val_size
        else:
            return self.test_size
        
    # [CLS]  [M]  w2  w3  [SEP]        
    # [CLS]  w1  [M]  w3  [SEP]        
    # [CLS]  w1 w2  [M]  [SEP]        
    # get the probability of [M] for each mask-prediction case
    # mask for (number of words) times and every time get (number of words + 2) probability
    # TODO: get the pred for only once
    def get_mask_pred_probs(self, seq1):
        mask_sentences = []
        
        for i in range(1, len(seq1) - 1):
            mask_seq = seq1.detach().clone()
            mask_seq[i] = self.MASK_token_id
            mask_sentences.append(mask_seq)

        mask_stack = torch.stack(mask_sentences)
        mask_stack = mask_stack.to(self.device)
        
        self.mask_predict_model.eval()

        with torch.no_grad():
            pred = self.mask_predict_model(mask_stack)[0]
        pred = pred.cpu()
        return pred
    
    def get_stack_probs(self, pred):
        bos_prob = torch.zeros(self.n_words)
        bos_prob[self.SOS_token_id] = 1

        eos_prob = torch.zeros(self.n_words)
        eos_prob[self.EOS_token_id] = 1

        mask_preds = []
        for idx in range(pred.shape[0]):
            mask_preds.append(pred[idx][idx+1])
        mask_stack = torch.stack(mask_preds)
        mask_stack = torch.cat((bos_prob.reshape(1,-1), mask_stack, eos_prob.reshape(1,-1)))
        return mask_stack
    
    def init_tokenizer(self, pretrained_model_name):
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        return tokenizer
    
    def init_constants(self):
        PAD_id,  SOS_id, EOS_id, UNK_id = self.tokenizer.convert_tokens_to_ids(["[PAD]", "[CLS]", "[SEP]", "[UNK]"])
        self.PAD_token_id = PAD_id
        self.SOS_token_id = SOS_id
        self.EOS_token_id = EOS_id
        self.UNK_token_id = UNK_id
        
        self.PAD_token = '[PAD]'
        self.SOS_token = '[CLS]'
        self.EOS_token = '[SEP]'
        self.UNK_token = '[UNK]'
        
        self.MASK_token = '[MASK]'
        self.MASK_token_id = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

        
    def read_text(self, text_path):
        # add words to dictionary
        f = open(text_path, 'r')
        lines = f.readlines()
        print(len(lines))
        if self.mode == "train":
            lines = lines[:self.train_size]
        elif self.mode == 'val':
            lines = lines[self.train_size:self.train_size+self.val_size]
        else:
            lines = lines[self.train_size+self.val_size:self.train_size+self.val_size+self.test_size]
        
        return lines



# In[11]:


# # # dry-run
# preprocessed_folder = '../../data/preprocess_quora_bert_mask_predict/'
# dataset = QuoraBertMaskPredictProbDataset("train", 10000, 100, text_path='../../data/quora_train.txt', preprocessed_folder=preprocessed_folder)


# In[16]:


# from tqdm import tqdm
# for i in tqdm(range(1000)):
#     _, __ , ___ = dataset[i]


# In[ ]:




