#!/usr/bin/env python
# coding: utf-8

# In[13]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import yaml
from argparse import ArgumentParser
import os

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from utils import cal_loss, cal_performance, log_performances, same_seeds


# In[ ]:


# parse argument
parser = ArgumentParser()
parser.add_argument("--config_path", dest="config_path",
                    default='../configs/dpng_transformer_bert_tokenizer_bow_indivtopk.yaml')
parser.add_argument("--preprocessed", dest="preprocessed", action="store_true")
parser.add_argument("--seed", dest="seed", default=0, type=int)

args = parser.parse_args()
config_path = args.config_path
preprocessed = args.preprocessed
seed = args.seed
print("config_path:", config_path)
print("preprocessed: ", preprocessed)
print("seed: ", seed)


# In[17]:


##### Read Arguments from Config File #####

# read from command line

# config_path = '../configs/base_transformer.yaml'
# config_path = '../configs/dpng_transformer.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indiv_neighbors.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_maskword_indivtopk.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_onlybow.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_replace.yaml'
# config_path='../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_replace_nopreprocess.yaml'
# config_path = '../configs/dpng_transformer_wordnet.yaml'
# config_path = '../configs/dpng_transformer_wordnet_replace_nopreprocess.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_replace_nopreprocess_no_append_bow.yaml'

# preprocessed = False

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

save_model_path = config['save_model_path']
log_file = config['log_file']
use_dataset = config['dataset']

num_epochs = config['num_epochs']
batch_size = config['batch_size']

d_model = config['d_model']
d_inner_hid = config['d_inner_hid']
d_k = config['d_k']
d_v = config['d_v']

n_head = config['n_head']
n_layers = config['n_layers']
n_warmup_steps = config['n_warmup_steps']

dropout = config['dropout']
embs_share_weight = config['embs_share_weight']
proj_share_weight = config['proj_share_weight']
label_smoothing = config['label_smoothing']

train_size = config['train_size']
val_size = config['val_size']

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
        append_bow = config['append_bow']
        
except KeyError:
    is_bow = False
    
try:
    use_wordnet = config['use_wordnet']
    indiv_k = config['indiv_k']
    replace_origin = config['replace_origin']
    append_bow = config['append_bow']
except KeyError:
    use_wordnet = False

# todo: add to params
lr = float(config['lr'])
# lr = 5e-4
# ###################


# In[ ]:


# debug
# batch_size = 50
# n_warmup_steps = 30000


# In[14]:


# same seed
# seed = 0
same_seeds(seed)

# set model and log path
seed_model_root = '../models/fixseed/seed{}/'.format(seed)
seed_log_root = '../logs/fixseed/seed{}/'.format(seed)

if not os.path.exists(seed_model_root):
    os.makedirs(seed_model_root)

if not os.path.exists(seed_log_root):
    os.makedirs(seed_log_root)

save_model_path = seed_model_root + save_model_path.split('/')[-1]
log_file = seed_log_root + log_file.split('/')[-1]
print('seed: ', seed)
print('save model path: ', save_model_path)
print('log path: ', log_file)


# In[ ]:


# load dataset
# preprocessed = False
if preprocessed:
    from datasets.quora_preprocessed_dataset import QuoraPreprocessedDataset as Dataset
else:
    if use_dataset == 'quora_dataset':
        from datasets.quora_dataset import QuoraDataset as Dataset
    elif use_dataset == 'quora_bert_dataset':
        from datasets.quora_bert_dataset import QuoraBertDataset as Dataset
    elif use_dataset == 'quora_bert_mask_predict_dataset':
        from datasets.quora_bert_mask_predict_dataset import QuoraBertMaskPredictDataset as Dataset
    elif use_dataset == 'quora_word_mask_prediction_dataset':
        from datasets.quora_word_mask_prediction_dataset import QuoraWordMaskPredictDataset as Dataset
    elif use_dataset == 'quora_wordnet_dataset':
        from datasets.quora_wordnet_dataset import QuoraWordnetDataset as Dataset
    elif use_dataset == 'quora_wordnet_aug_dataset':
        from datasets.quora_wordnet_aug_dataset import QuoraWordnetAugDataset as Dataset        
    else:
        raise NotImplementedError("Dataset is not defined or not implemented")


# In[ ]:


def create_mini_batch(samples):
    seq1_tensors = [s[0] for s in samples]
    seq2_tensors = [s[1] for s in samples]

    # zero pad
    seq1_tensors = pad_sequence(seq1_tensors,
                                  batch_first=True)

    seq2_tensors = pad_sequence(seq2_tensors,
                                  batch_first=True)    
    
    return seq1_tensors, seq2_tensors


# In[ ]:


if preprocessed:
    model_name = config_path.split('/')[-1][:-5]
    preprocessed_file = '../data/preprocess_all_{}.npy'.format(model_name)
    dataset = Dataset("train", train_size, val_size, preprocessed_file=preprocessed_file)
    val_dataset = Dataset("val", train_size, val_size, preprocessed_file=preprocessed_file)    
elif is_bow:
    dataset = Dataset(
        "train", train_size, val_size, bow_strategy=bow_strategy, topk=topk, indiv_topk=indiv_topk, 
        only_bow=only_bow, use_origin=only_bow, replace_predict=replace_predict, append_bow=append_bow
    )
    # try not to replace predict when validation?
    val_dataset = Dataset(
        "val", train_size, val_size, bow_strategy=bow_strategy, topk=topk, indiv_topk=indiv_topk, 
        only_bow=only_bow, use_origin=only_bow, replace_predict=False, append_bow=append_bow
    )
elif use_wordnet:
    dataset = Dataset("train", train_size, val_size, indiv_k=indiv_k, replace_origin=replace_origin, append_bow=append_bow)
    val_dataset = Dataset("val", train_size, val_size, indiv_k=indiv_k, replace_origin=replace_origin, append_bow=append_bow)
else:
    dataset = Dataset("train", train_size, val_size)
    val_dataset = Dataset("val", train_size, val_size)

data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=False)


# In[ ]:


transformer = Transformer(
    dataset.n_words,
    dataset.n_words,
    src_pad_idx=dataset.PAD_token_id,
    trg_pad_idx=dataset.PAD_token_id,
    trg_emb_prj_weight_sharing=proj_share_weight,
    emb_src_trg_weight_sharing=embs_share_weight,
    d_k=d_k,
    d_v=d_v,
    d_model=d_model,
    d_word_vec=d_model,
    d_inner=d_inner_hid,
    n_layers=n_layers,
    n_head=n_head,
    dropout=dropout,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer.to(device)


# In[ ]:


optimizer = ScheduledOptim(
    optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=lr),
    2.0, d_model, n_warmup_steps)
# optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=lr)


# In[ ]:


# train epoch
def train_epoch(model, data_loader, optimizer, device, smoothing=False):
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 

    for seq1, seq2 in tqdm(data_loader):
        src_seq = seq1.to(device)
        trg_seq = seq2[:, :-1].to(device)
        gold = seq2[:, 1:].contiguous().view(-1).to(device)

        optimizer.zero_grad()
        try:
            pred = model(src_seq, trg_seq)
        except RuntimeError as e:
#             print(src_seq, trg_seq)
            print("[Info] Length of src seq: {}, trg seq: {}".format(len(src_seq), len(trg_seq)))
            print(e)
            # sentence too long, skip the training batch
            continue
#             raise RuntimeError(e)
            
        try:
            loss, n_correct, n_word = cal_performance(
                pred, gold, dataset.PAD_token_id, smoothing) 
            loss.backward()
    #         optimizer.step()
            optimizer.step_and_update_lr()
        # CUDA out of memory
        except RuntimeError as e:
            print("[Info] Length of src seq: {}, trg seq: {}".format(len(src_seq), len(trg_seq)))
            print(e)
            # sentence too long, skip the training batch
            continue

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    
    return loss_per_word, accuracy


# In[ ]:


def eval_epoch(model, val_data_loader, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    with torch.no_grad():
        for seq1, seq2 in tqdm(val_data_loader):

            src_seq = seq1.to(device)
            trg_seq = seq2[:, :-1].to(device)
            gold = seq2[:, 1:].contiguous().view(-1).to(device)

            pred = model(src_seq, trg_seq)

            loss, n_correct, n_word = cal_performance(
                pred, gold, dataset.PAD_token_id, smoothing=False) 

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


# In[ ]:


# # reproduce
# import random
# random.seed(0)
# torch.manual_seed(0)


# In[ ]:


f = open(log_file, 'w')
# f = open('../logs/tmp.txt', 'w')

best_loss = 999

f.write("Config: {}\n".format(config))

for epoch in range(num_epochs):
    print("Epoch {} / {}".format(epoch + 1, num_epochs))
    start = time.time()
    train_loss, train_accu = train_epoch(model, data_loader, optimizer, device, smoothing=label_smoothing)
    log_performances('Training', train_loss, train_accu, start, f)

    start = time.time()
    valid_loss, valid_accu = eval_epoch(model, val_data_loader, device)
    log_performances('Validation', valid_loss, valid_accu, start, f)
    
    if valid_loss < best_loss:
        # save model
        torch.save(model.state_dict(), save_model_path)
        best_loss = valid_loss
        print("model saved in Epoch {}".format(epoch + 1))
        f.write("model saved in Epoch {}\n".format(epoch + 1))
        f.flush()

f.close()
    


# In[ ]:




