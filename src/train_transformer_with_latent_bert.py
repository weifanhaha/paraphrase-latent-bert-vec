#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import yaml
from argparse import ArgumentParser


# from transformer.Models import Transformer, Encoder, Decoder
from datasets.quora_bert_mask_predict_prob_dataset import QuoraBertMaskPredictProbDataset
from transformer.CustomModels import Transformer

from transformer.Optim import ScheduledOptim


# In[2]:


# parse argument
parser = ArgumentParser()
parser.add_argument("--config_path", dest="config_path",
                    default='../configs/dpng_transformer_bert_attention.yaml')
parser.add_argument("--bert_alpha", dest="bert_alpha", type=float, default=0)

args = parser.parse_args()
config_path = args.config_path
bert_alpha = args.bert_alpha
print("config_path:", config_path)
print("bert_alpha: ", bert_alpha)


# In[2]:


# bert_alpha = 0 # normal
# bert_alpha = 0.5 # half-half
# bert_alpha = 1 # only bert attention


# In[2]:


##### Read Arguments from Config File #####

config_path = '../configs/dpng_transformer_bert_attention.yaml'

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)

save_model_path = config['save_model_path']
log_file = config['log_file']
preprocessed_folder = config['preprocessed_folder']

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
# ###################


# In[3]:


def create_mini_batch(samples):
    seq1_tensors = [s[0] for s in samples]
    seq2_tensors = [s[1] for s in samples]
    probs_tensors = [s[2] for s in samples]

    # zero pad
    seq1_tensors = pad_sequence(seq1_tensors,
                                  batch_first=True)

    seq2_tensors = pad_sequence(seq2_tensors,
                                  batch_first=True)
    probs_tensors = pad_sequence(probs_tensors,
                                  batch_first=True)
    
    return seq1_tensors, seq2_tensors, probs_tensors

dataset = QuoraBertMaskPredictProbDataset("train", train_size, val_size, text_path='../data/quora_train.txt', preprocessed_folder=preprocessed_folder)
val_dataset = QuoraBertMaskPredictProbDataset("val", train_size, val_size, text_path='../data/quora_train.txt', preprocessed_folder=preprocessed_folder)

data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=False)


# In[4]:


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
    bert_alpha=bert_alpha
)


# In[8]:


optimizer = ScheduledOptim(
    optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
    2.0, d_model, n_warmup_steps)


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer.to(device)


# In[12]:


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss



# In[13]:


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


# In[14]:


# train epoch
def train_epoch(model, data_loader, optimizer, device, smoothing=False):
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 
    
    trange = tqdm(data_loader)

    for seq1, seq2, probs, in trange:
        src_seq = seq1.to(device)
        trg_seq = seq2[:, :-1].to(device)
        probs = probs.to(device)
        gold = seq2[:, 1:].contiguous().view(-1).to(device)

        optimizer.zero_grad()
        pred = model(src_seq, trg_seq, probs)

        loss, n_correct, n_word = cal_performance(
            pred, gold, dataset.PAD_token_id, smoothing) 
        loss.backward()
        optimizer.step_and_update_lr()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
        trange.set_postfix({
            'loss': loss.item(),
            'acc': n_correct/n_word
        })

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy



# In[15]:


def eval_epoch(model, val_data_loader, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    with torch.no_grad():
        for seq1, seq2, probs in tqdm(val_data_loader):

            src_seq = seq1.to(device)
            trg_seq = seq2[:, :-1].to(device)
            probs = probs.to(device)
            gold = seq2[:, 1:].contiguous().view(-1).to(device)

            pred = model(src_seq, trg_seq, probs)

            loss, n_correct, n_word = cal_performance(
                pred, gold, dataset.PAD_token_id, smoothing=False) 

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy



# In[16]:


def log_performances(header, loss, accu, start_time, file):
    print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '        'elapse: {elapse:3.3f} sec'.format(
            header=f"({header})", ppl=math.exp(min(loss, 100)),
            accu=100*accu, elapse=(time.time()-start_time))) 
    file.write(
          '- {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
        'elapse: {elapse:3.3f} sec\n'.format(
            header=f"({header})", ppl=math.exp(min(loss, 100)),
            accu=100*accu, elapse=(time.time()-start_time))
    )


# In[21]:


# debug
# log_file = "../logs/tmp.txt"
f = open(log_file, 'w')
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




