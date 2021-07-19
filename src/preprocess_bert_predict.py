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
import yaml
from argparse import ArgumentParser

from transformers import BertTokenizer, BertForMaskedLM
from datasets.quora_bert_mask_predict_dataset import QuoraBertMaskPredictDataset as Dataset
from utils import same_seeds


# In[1]:


# parse argument
parser = ArgumentParser()
parser.add_argument("--config_path", dest="config_path",
                    default='../configs/dpng_transformer_bert_tokenizer_bow_indivtopk.yaml')
parser.add_argument("--seed", dest="seed", default=0, type=int)

args = parser.parse_args()
config_path = args.config_path
seed = args.seed
print("config_path:", config_path)
print("seed: ", seed)

same_seeds(seed)


# In[9]:


# config_path = '../configs/dpng_transformer_bert_tokenizer_bow.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_onlybow.yaml'


with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    
train_size = config['train_size']
val_size = config['val_size']
test_size = config['test_size']

batch_size = 128

is_bow = config['is_bow']
bow_strategy = config['bow_strategy']
topk = config['topk']
if bow_strategy != 'simple_sum':
    indiv_topk = config['indiv_topk']
else:
    # not used but use default value for simplicity
    indiv_topk = 50


try:
    is_bow = config['is_bow']

    if is_bow:
        bow_strategy = config['bow_strategy']
        topk = config['topk']
        if bow_strategy != 'simple_sum':
            indiv_topk = config['indiv_topk']
        else:
            # not used but use default value for simplicity
            indiv_topk = 50
        
        only_bow = config['only_bow']
        replace_predict = config['replace_predict']


except KeyError:
    is_bow = False


# In[4]:


model_name = config_path.split('/')[-1][:-5]
output_all_path = '../data/preprocess_all_{}.npy'.format(model_name)


# In[6]:


# just init one dataset / bert model
dataset = Dataset(
    "train", train_size+val_size+test_size, 0, 0, bow_strategy=bow_strategy, topk=topk, indiv_topk=indiv_topk,
    only_bow=only_bow, use_origin=only_bow, replace_predict=replace_predict
)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


# In[7]:


sentence_tensors = []


# In[8]:


for seq1_tensor, seq2_tensor in tqdm(data_loader):
    seq1_tensor = seq1_tensor.view(-1)
    seq2_tensor = seq2_tensor.view(-1)
    sentence_tensors.append([seq1_tensor, seq2_tensor])


# In[9]:


arr = np.array(sentence_tensors, dtype=object)
np.save(output_all_path, arr)

# load_arr = np.load(output_all_path, allow_pickle=True)


# In[ ]:




