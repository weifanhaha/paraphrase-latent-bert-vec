#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import torch
import torch.nn.functional as F
import time
import numpy as np


# In[1]:


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


# In[2]:


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


# In[ ]:

# perplexity is exp(cross entropy loss)
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

# perplexity is exp(cross entropy loss)
def log_performances_with_cls(header, loss, accu, cls_loss, avg_loss, start_time, file):
    log_tmp = '  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                   'avg_cls_loss: {avg_cls_loss: 2.5f}, avg_loss: {avg_loss: 8.5f}, '\
                   'elapse: {elapse:3.3f} sec\n'
    
    print(
        log_tmp.format(header=f"({header})", ppl=math.exp(min(loss, 100)),
        accu=100*accu, avg_cls_loss=cls_loss, avg_loss=avg_loss, elapse=(time.time()-start_time))
    ) 

    
    file.write(
          log_tmp.format(header=f"({header})", ppl=math.exp(min(loss, 100)),
          accu=100*accu, avg_cls_loss=cls_loss, avg_loss=avg_loss, elapse=(time.time()-start_time))
    )

    
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    
