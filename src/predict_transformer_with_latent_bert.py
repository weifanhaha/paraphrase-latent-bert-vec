#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import argparse
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import yaml
from argparse import ArgumentParser

from transformer.CustomModels import Transformer, Translator
from datasets.quora_bert_mask_predict_prob_dataset import QuoraBertMaskPredictProbDataset
from datasets.quora_text_dataset import QuoraTextDataset


# In[2]:


##### Read Arguments from Config File #####

config_path = '../configs/dpng_transformer_bert_attention.yaml'

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)

save_model_path = config['save_model_path']
save_model_path = '../models/tune/transformer_bert_enc_attention_alpha0.5.pth'
log_file = config['log_file']
output_file = config['test_output_file']
preprocessed_folder = config['preprocessed_folder']

num_epochs = config['num_epochs']


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

beam_size = 3
max_seq_len = 30
batch_size = 1

# ###################


# In[3]:


bert_alpha = 0.5 # half-half


# In[4]:


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


# In[5]:


dataset = QuoraBertMaskPredictProbDataset("test", train_size, val_size, test_size, preprocessed_folder=preprocessed_folder)
# dataset = QuoraBertMaskPredictProbDataset("test", train_size, val_size, test_size)
text_dataset = QuoraTextDataset("test", train_size, val_size, test_size)
data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=False)


# In[6]:


# text_dataset[0]
i = 5
idxs, idxs2, probs = dataset[i]
print(text_dataset[i])
print(dataset.tokenizer.convert_ids_to_tokens(idxs))
print(dataset.tokenizer.convert_ids_to_tokens(idxs2))


# In[7]:


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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



model = transformer.to(device)

model.load_state_dict((torch.load(
        save_model_path, map_location=device)))


# In[9]:


src_pad_idx = dataset.PAD_token_id
trg_pad_idx = dataset.PAD_token_id
    
trg_bos_idx = dataset.SOS_token_id
trg_eos_idx = dataset.EOS_token_id
unk_idx = dataset.UNK_token_id


# In[10]:


translator = Translator(
        model=model,
        beam_size=beam_size,
        max_seq_len=max_seq_len,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_bos_idx=trg_bos_idx,
        trg_eos_idx=trg_eos_idx,
).to(device)


# In[11]:


# dry-run
seq1, seq2, bert_prob = next(iter(data_loader))
input_seq = seq1.to(device)
bert_prob = bert_prob.to(device)
pred_seq = translator.translate_sentence(input_seq, bert_prob)
print(dataset.tokenizer.convert_ids_to_tokens(pred_seq))


# In[14]:


tokenizer = dataset.tokenizer
# count = 0
with open(output_file, 'w') as f:
    for i, (seq1, seq2, bert_prob) in enumerate(tqdm(data_loader)):
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




