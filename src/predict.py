#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import argparse
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import yaml
import os

from transformer.Models import Transformer
from transformer.Translator import Translator
from utils import same_seeds


# In[4]:


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


# In[ ]:


# fix seed
same_seeds(seed)


# In[5]:


##### Read Arguments from Config File #####

# config_path = '../configs/base_transformer.yaml'
# config_path = '../configs/dpng_transformer.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_onlybow.yaml'
# config_path = '../configs/dpng_transformer_wordnet.yaml'

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

save_model_path = config['save_model_path']
output_file = config['test_output_file']
use_dataset = config['dataset']

batch_size = 1

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

try:
    is_bow = config['is_bow']

    if is_bow:
        bow_strategy = config['bow_strategy']
        topk = config['topk']
        only_bow = config['only_bow']
        if bow_strategy != 'simple_sum':
            indiv_topk = config['indiv_topk']
        else:
            # not used but use default value for simplicity
            indiv_topk = 50

except KeyError:
    is_bow = False
    
try:
    use_wordnet = config['use_wordnet']
    indiv_k = config['indiv_k']
    replace_origin = config['replace_origin']
    append_bow = config['append_bow']
except KeyError:
    use_wordnet = False
    
# ###################


# In[ ]:


# set model and log path
seed_model_root = '../models/fixseed/seed{}/'.format(seed)
seed_output_root = '../outputs/fixseed/seed{}/'.format(seed)

if not os.path.exists(seed_model_root):
    os.makedirs(seed_model_root)

if not os.path.exists(seed_output_root):
    os.makedirs(seed_output_root)

save_model_path = seed_model_root + save_model_path.split('/')[-1]
output_file = seed_output_root + output_file.split('/')[-1]
print('seed: ', seed)
print('save model path: ', save_model_path)
print('output file: ', output_file)


# In[3]:


# ############### Arguments ###############
# # The argument is same for DNPG paper
# save_model_path = '../models/DNPG_base_transformer.pth'
# output_file = '../outputs/test_DNPG_transformer_out.txt'
# max_seq_len = 30

# batch_size = 1
# beam_size = 3 # 1 for greedy

# d_model = 450
# d_inner_hid = 512
# d_k = 64
# d_v = 64

# n_head = 9
# n_layers = 3
# dropout = 0.1
# embs_share_weight = True
# proj_share_weight = True

# train_size = 100000
# val_size = 4000
# test_size = 20000
# #######################################


# In[4]:


# ##### Arguments #####
# d_model = 512
# save_model_path = '../models/base_transformer.pth'
# batch_size = 1
# beam_size = 3 # 1 for greedy
# max_seq_len = 30
# output_file = '../outputs/val_transformer_out.txt'

# d_model = 512
# d_inner_hid = 512
# d_k = 64
# d_v = 64

# n_head = 8
# n_layers = 6
# dropout = 0.1
# embs_share_weight = True
# proj_share_weight = True
# ###################


# In[6]:


# load dataset
from datasets.quora_text_dataset import QuoraTextDataset

if use_dataset == 'quora_dataset':
    from datasets.quora_dataset import QuoraDataset as Dataset
elif use_dataset == 'quora_wordnet_dataset':
    from datasets.quora_wordnet_dataset import QuoraWordnetDataset as Dataset
elif use_dataset == 'quora_wordnet_aug_dataset':
    from datasets.quora_wordnet_aug_dataset import QuoraWordnetAugDataset as Dataset        
else:
    raise NotImplementedError("Dataset is not defined or not implemented: {}".format(use_dataset))


# In[7]:


def create_mini_batch(samples):
    seq1_tensors = [s[0] for s in samples]
    seq2_tensors = [s[1] for s in samples]

    # zero pad
    seq1_tensors = pad_sequence(seq1_tensors,
                                  batch_first=True)

    seq2_tensors = pad_sequence(seq2_tensors,
                                  batch_first=True)    
    
    return seq1_tensors, seq2_tensors

if use_wordnet:
    dataset = Dataset("test", train_size, val_size, test_size, indiv_k=indiv_k, replace_origin=False, append_bow=append_bow)
else:
    dataset = Dataset("test", train_size, val_size, test_size)
same_seeds(seed)    
text_dataset = QuoraTextDataset("test", train_size, val_size, test_size)
data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=False)


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    dropout=dropout    
)

model = transformer.to(device)

model.load_state_dict((torch.load(
        save_model_path, map_location=device)))


# In[ ]:


src_pad_idx = dataset.PAD_token_id
trg_pad_idx = dataset.PAD_token_id
    
trg_bos_idx = dataset.SOS_token_id
trg_eos_idx = dataset.EOS_token_id
unk_idx = dataset.UNK_token_id


# In[ ]:


translator = Translator(
        model=model,
        beam_size=beam_size,
        max_seq_len=max_seq_len,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_bos_idx=trg_bos_idx,
        trg_eos_idx=trg_eos_idx).to(device)


# In[8]:


# todo: modify to  Source / Target / Input / Output
idx2word = dataset.idx2word
with open(output_file, 'w') as f:
    for i, (seq1, seq2) in enumerate(tqdm(data_loader)):
        input_seq = seq1.to(device)
        trg_seq = seq2[0].tolist()
        
        pred_seq = translator.translate_sentence(input_seq)
        pred_line = ' '.join(idx2word[idx] for idx in pred_seq)
        pred_line = pred_line.replace(dataset.SOS_token, '').replace(dataset.EOS_token, '')
            
        input_line = ' '.join(idx2word[idx] for idx in input_seq[0].tolist())
        input_line = input_line.replace(dataset.SOS_token, '').replace(dataset.EOS_token, '')

        src_line, trg_line = text_dataset.sentences[i]
        
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


# In[33]:





# In[ ]:




