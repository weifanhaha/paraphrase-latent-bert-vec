#!/usr/bin/env python
# coding: utf-8

# In[14]:


from transformer.Models import Transformer


# In[15]:


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
# from datasets.quora_bert_mask_predict_prob_dataset import QuoraBertMaskPredictProbDataset
from transformer.Optim import ScheduledOptim


# In[16]:


from nltk.corpus import stopwords
from nltk.corpus import wordnet 
import random


# In[54]:


from tqdm import tqdm
import re
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import os
import random

class QuoraBertMaskPredictProbDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, 
                 text_path='../data/quora_train.txt', pretrained_model_name='bert-base-cased', preprocessed_folder=None, replace_prob=0.5):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        
        self.tokenizer = self.init_tokenizer(pretrained_model_name)
        self.mask_predict_model = BertForMaskedLM.from_pretrained(pretrained_model_name)
        self.sentences = self.read_text(text_path)
        self.init_constants()
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.n_words = len(self.tokenizer)
        self.stopwords = stopwords.words('english')
        self.replace_prob = replace_prob
        
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
        
        aug_seq = self._get_aug_sentence(seq2)
        tokens2 = self.tokenizer.tokenize(aug_seq)
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
    
    def _get_aug_sentence(self, seq):
        words = seq.split()
        new_sentence = []
        
        for word in words:
            syn_set = self._get_synset(word)
            
            if word in self.stopwords or len(syn_set) == 0:
                new_sentence.append(word)
                continue

            # replace with prob 
            if random.random() > self.replace_prob:
                new_sentence.append(word)
                continue
            
            sampled_syn = random.sample(syn_set, 1)[0]
            new_sentence.append(sampled_syn)
                
        return ' '.join(new_sentence)
    
    
    def _get_synset(self, word):
        syn_set = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas(): 
                # ignore the phrase
                if '_' in l.name() or '-' in l.name():
                    continue
                syn_set.add(l.name()) 
        if word in syn_set:
            syn_set.remove(word)
        return syn_set

    
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
        # shuffle
        np.random.shuffle(lines)
        if self.mode == "train":
            lines = lines[:self.train_size]
        elif self.mode == 'val':
            lines = lines[self.train_size:self.train_size+self.val_size]
        else:
            lines = lines[self.train_size+self.val_size:self.train_size+self.val_size+self.test_size]
        
        return lines




# In[55]:


seed = 33333

# fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(seed)


# In[56]:


##### Read Arguments from Config File #####

config_path = '../../configs/dpng_transformer_bert_attention_wordnet_aug.yaml'
# config_path = '../../configs/dpng_transformer_bert_attention_more_epochs.yaml'


with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)

save_model_path = config['save_model_path']
log_file = config['log_file']
output_file = config['test_output_file']
preprocessed_folder = config['preprocessed_folder']

# save_model_path = '../../models/tune/transformer_key_enc_bert_val_attention_alpha0.5_epoch100.pth'
# log_file = '../../logs/tune/transformer_key_enc_bert_val_attention_alpha0.5_epoch100.txt'
# output_file = '../../outputs/tune/transformer_key_enc_bert_val_attention_alpha0.5_epoch100.txt'


# set model and log path
seed_model_root = '../../models/fixseed/seed{}/'.format(seed)
seed_log_root = '../../logs/fixseed/seed{}/'.format(seed)
seed_output_root = '../../outputs/fixseed/seed{}/'.format(seed)

if not os.path.exists(seed_model_root):
    os.makedirs(seed_model_root)

if not os.path.exists(seed_log_root):
    os.makedirs(seed_log_root)

save_model_path = seed_model_root + save_model_path.split('/')[-1]
log_file = seed_log_root + log_file.split('/')[-1]
output_file = seed_output_root + output_file.split('/')[-1]
print('seed: ', seed)
print('save model path: ', save_model_path)
print('log path: ', log_file)
print('output path: ', output_file)

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
test_size = config['test_size']


# ###################


# In[57]:


# bert_alpha = 0 # normal
bert_alpha = 0.5 # half-half
# bert_alpha = 1 # only bert attention


# In[58]:


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

dataset = QuoraBertMaskPredictProbDataset("train", train_size, val_size, text_path='../../data/quora_train.txt')
val_dataset = QuoraBertMaskPredictProbDataset("val", train_size, val_size, text_path='../../data/quora_train.txt')

data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=False)


# In[62]:


# i = 6
# for _ in range(3):
#     print(dataset.tokenizer.convert_ids_to_tokens(dataset[i][1]))
    


# In[7]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from transformer.Models import get_pad_mask, get_subsequent_mask
from transformer.Models import PositionalEncoding, Encoder

# QKV: dec_out  enc_out  bert_out
class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, bert_alpha=0):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.bert_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.bert_alpha = bert_alpha

    def forward(
            self, dec_input, enc_output, bert_out,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        
        dec_bert_output, dec_enc_bert_attn = self.bert_attn(
            dec_output, enc_output, bert_out, mask=dec_enc_attn_mask)
        
        dec_output = (1-self.bert_alpha) *  dec_output + self.bert_alpha * dec_bert_output
        
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn



# In[8]:



class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, bert_alpha=0):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, bert_alpha=bert_alpha)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, bert_out, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))
        dec_output = self.layer_norm(dec_output)
        
#         print('encoder output', enc_output.shape)
#         print('decoder output', dec_output.shape)
#         print('bert output', bert_out.shape)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, bert_out, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


# In[9]:


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True, bert_alpha=0):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        
        self.probnn = nn.Linear(n_src_vocab, d_model)

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, bert_alpha=bert_alpha)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec,         'To facilitate the residual connections,          the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq, bert_prob, return_attns=False):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, enc_slf_attn_list = self.encoder(src_seq, src_mask, return_attns=True)
        
        bert_out = self.probnn(bert_prob)
#         print("bert prod:", bert_prob.shape)
#         print("(trans) bert out: ", bert_out.shape)
        
        dec_output, dec_slf_attn_list, dec_enc_attn_list = self.decoder(trg_seq, trg_mask, enc_output, bert_out, src_mask, return_attns=True)
        seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale

        if return_attns:
            return seq_logit.view(-1, seq_logit.size(2)), enc_slf_attn_list, dec_slf_attn_list, dec_enc_attn_list
        return seq_logit.view(-1, seq_logit.size(2))





# In[10]:


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


# In[11]:


optimizer = ScheduledOptim(
    optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
    2.0, d_model, n_warmup_steps)


# In[12]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer.to(device)


# In[13]:


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




# In[14]:


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


# In[15]:


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
        try:
            pred = model(src_seq, trg_seq, probs)
            loss, n_correct, n_word = cal_performance(
                pred, gold, dataset.PAD_token_id, smoothing) 
            loss.backward()
            optimizer.step_and_update_lr()
        # cuda out of memory
        except RuntimeError as e:
            print("[Info] Length of src seq: {}, trg seq: {}".format(len(src_seq), len(trg_seq)))
            print(e)
            # sentence too long, skip the training batch
            continue
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




# In[16]:


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




# In[17]:


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


# In[18]:


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


# In[15]:


from transformer.CustomModels import Translator


# In[16]:


import re
import numpy as np

class QuoraTextDataset(Dataset):
    def __init__(self, mode, train_size=5000, val_size=1000, test_size=1000, text_path='../data/quora_train.txt'):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.sentences = []

        self._init_sentences(text_path)

    def __getitem__(self, idx):
        return self.sentences[idx]
        seq1, seq2 = self.sentences[idx]

    def __len__(self):
        if self.mode == 'train':
            return self.train_size
        elif self.mode == 'val':
            return self.val_size
        else:
            return self.test_size
    
    def _init_sentences(self, text_path):
        f = open(text_path, 'r')
        lines = f.readlines()
        np.random.shuffle(lines)
        if self.mode == "train":
            lines = lines[:self.train_size]
        elif self.mode == 'val':
            lines = lines[self.train_size:self.train_size+self.val_size]
        else:
            lines = lines[self.train_size+self.val_size:self.train_size+self.val_size+self.test_size]
        
        def normalize_sentence(s):
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
            return s

        for line in lines:
            seq1, seq2 = [normalize_sentence(seq) for seq in line.split('\t')]
            self.sentences.append((seq1, seq2))

        


# In[20]:


tbatch_size = 1
beam_size = 3
max_seq_len = 30

same_seeds(seed)

# dataset = QuoraBertMaskPredictProbDataset("test", train_size, val_size, test_size, preprocessed_folder=preprocessed_folder)
test_dataset = QuoraBertMaskPredictProbDataset("test", train_size, val_size, test_size, text_path='../../data/quora_train.txt')

same_seeds(seed)
text_dataset = QuoraTextDataset("test", train_size, val_size, test_size, text_path='../../data/quora_train.txt')
test_data_loader = DataLoader(test_dataset, batch_size=tbatch_size, collate_fn=create_mini_batch, shuffle=False)


# In[21]:


model.load_state_dict((torch.load(
        save_model_path, map_location=device)))


# In[22]:


src_pad_idx = dataset.PAD_token_id
trg_pad_idx = dataset.PAD_token_id
    
trg_bos_idx = dataset.SOS_token_id
trg_eos_idx = dataset.EOS_token_id
unk_idx = dataset.UNK_token_id


# In[23]:


translator = Translator(
        model=model,
        beam_size=beam_size,
        max_seq_len=max_seq_len,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_bos_idx=trg_bos_idx,
        trg_eos_idx=trg_eos_idx,
).to(device)


# In[25]:


# dry-run
seq1, seq2, bert_prob = next(iter(test_data_loader))
input_seq = seq1.to(device)
bert_prob = bert_prob.to(device)
pred_seq = translator.translate_sentence(input_seq, bert_prob)
print(dataset.tokenizer.convert_ids_to_tokens(pred_seq))


# In[26]:


tokenizer = dataset.tokenizer
# count = 0
with open(output_file, 'w') as f:
    for i, (seq1, seq2, bert_prob) in enumerate(tqdm(test_data_loader)):
        src_line, trg_line = text_dataset.sentences[i]

        input_seq = seq1.to(device)
        bert_prob = bert_prob.to(device)
        trg_seq = seq2[0].tolist()

        pred_seq = translator.translate_sentence(input_seq, bert_prob)
        pred_tokens = tokenizer.convert_ids_to_tokens(pred_seq)
        pred_line = tokenizer.convert_tokens_to_string(pred_tokens)
        pred_line = pred_line.replace(dataset.SOS_token, '').replace(dataset.EOS_token, '')

        input_tokens = tokenizer.convert_ids_to_tokens(input_seq[0].tolist())
        input_line = tokenizer.convert_tokens_to_string(input_tokens)

        trg_tokens = tokenizer.convert_ids_to_tokens(trg_seq)
        trg_line = tokenizer.convert_tokens_to_string(trg_tokens)
        trg_line = trg_line.replace(dataset.SOS_token, '').replace(dataset.EOS_token, '')


#         print('*' * 80)
#         print('Source: ', src_line.strip())
#         print('Target: ', trg_line.strip())
#         print('Input: ', input_line.strip())
#         print('Predict: ', pred_line.strip())
        
        f.write('*' * 80)
        f.write('\n')
        f.write('Source: {}\n'.format(src_line.strip()))
        f.write('Target: {}\n'.format(trg_line.strip()))
        f.write('Input: {}\n'.format(input_line.strip()))
        f.write('Predict: {}\n'.format(pred_line.strip()))
        f.flush()

#         count += 1
#         if count > 5:
#             break


# In[ ]:




