#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM


# In[4]:


class QuoraWordMaskPredictDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, 
                 text_path='../data/quora_train.txt', pretrained_model_name="bert-large-cased-whole-word-masking", 
                 topk=50, bow_strategy='simple_sum', indiv_topk=10, indiv_topp=0.01):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.topk = topk
        self.bow_strategy = bow_strategy # simple_sum, mask_sum, indiv_topk, indiv_topp, indiv_neighbors
        self.indiv_topk = indiv_topk
        self.indiv_topp = indiv_topp
        
        self.tokenizer = self.init_tokenizer(pretrained_model_name)
        self.mask_predict_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name)
        self.sentences = self.read_text(text_path)
        self.init_constants()
        
        self.n_words = len(self.tokenizer)
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
        
        # pass the string sentence to mask words
        seq1_predict_token_tensors = self.get_source_predict_tokens(seq1)

        concat_tensor = torch.cat((seq1_tensor, seq1_predict_token_tensors.cpu()))
        concat_tensor = torch.cat((concat_tensor, torch.tensor([self.EOS_token_id])))

        return concat_tensor, seq2_tensor

    def __len__(self):
        if self.mode == 'train':
            return self.train_size
        elif self.mode == 'val':
            return self.val_size
        else:
            return self.test_size
        
    def get_source_predict_tokens(self, seq1):
        mask_sentences = []
        
        seq1_words = seq1.split()
        
        for i in range(len(seq1_words)):
            word = seq1_words[i]
            seq1_words[i] = self.MASK_token
            sentence = ' '.join(seq1_words)
            tokens = self.tokenizer.tokenize(sentence)
            word_pieces =  [self.SOS_token] + tokens + [self.EOS_token]
            idxes = self.tokenizer.convert_tokens_to_ids(word_pieces)
            mask_sentences.append(torch.tensor(idxes, dtype=torch.long))
            seq1_words[i] = word

        mask_stack = pad_sequence(mask_sentences, batch_first=True)
        
        masks_tensors = torch.zeros(mask_stack.shape,
                                    dtype=torch.long)
        # let bert attends only not padding ones
        masks_tensors = masks_tensors.masked_fill(
            mask_stack != 0, 1)

        mask_stack = mask_stack.to(self.device)
        masks_tensors = masks_tensors.to(self.device)
        self.mask_predict_model.eval()

        with torch.no_grad():
            pred = self.mask_predict_model(mask_stack, attention_mask=masks_tensors)[0]
        pred = pred.cpu()
            
        if self.bow_strategy == 'simple_sum':
            bows = torch.zeros(self.n_words)
            for i in range(pred.shape[0]):
                prob = pred[i][i+1]
                bows += prob
            _, indices = torch.topk(bows, self.topk)
            return indices
        elif self.bow_strategy == 'indiv_topk':
            # todo: try to improve efficiency with matrix calculation
            probs, indiv_indices = torch.topk(pred, self.indiv_topk)
            bows = torch.zeros(self.n_words)
            for i in range(indiv_indices.shape[0]):
                prob, indices = probs[i][i+1], indiv_indices[i][i+1]
                res = torch.zeros(self.n_words)
                res = res.scatter(0, indices, prob)
                bows += res
            _, indices = torch.topk(bows, self.topk)
            return indices    
        elif self.bow_strategy == 'indiv_neighbors':
            probs, indiv_indices = torch.topk(pred, self.indiv_topk)
            final_indices = []
            for i in range(indiv_indices.shape[0]):
                _, indices = probs[i][i+1], indiv_indices[i][i+1]
                final_indices.append(indices)
            
            return torch.cat(final_indices)
            
        else:
            raise ValueError("bow strategy is not defined")

    def init_tokenizer(self, pretrained_model_name):
        tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking")  
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
        if self.mode == "train":
            lines = lines[:self.train_size]
        elif self.mode == 'val':
            lines = lines[self.train_size:self.train_size+self.val_size]
        else:
            lines = lines[self.train_size+self.val_size:self.train_size+self.val_size+self.test_size]
        
        return lines



# In[5]:


# dataset = QuoraWordMaskPredictDataset("train", 1000, 100, text_path='../../data/quora_train.txt')


# In[7]:


# def create_mini_batch(samples):
#     seq1_tensors = [s[0] for s in samples]
#     seq2_tensors = [s[1] for s in samples]

#     # zero pad
#     seq1_tensors = pad_sequence(seq1_tensors,
#                                   batch_first=True)

#     seq2_tensors = pad_sequence(seq2_tensors,
#                                   batch_first=True)    
    
#     return seq1_tensors, seq2_tensors


# # it takes time to predict masked component
# # to improve -> use gpu and calculate outside the dataset
# data_loader = DataLoader(dataset, batch_size=64, collate_fn=create_mini_batch)


# In[8]:


# from tqdm import tqdm
# for seq1, seq2 in tqdm(data_loader):
# #     pass
# #     print(seq1)
#     print(seq1.shape, seq2.shape)


# In[ ]:




