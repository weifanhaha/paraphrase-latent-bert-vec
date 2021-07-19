#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch


# In[1]:


def get_bow(bow_strategy, n_words, pred, topk, indiv_topk):
    if bow_strategy == 'simple_sum':
        return get_simple_sum_bow(n_words, pred, topk)

    elif bow_strategy == 'indiv_topk':
        return get_indiv_topk_bow(n_words, pred, topk, indiv_topk)
        
    elif bow_strategy == 'indiv_topk':
        return get_indiv_neighbors_bow(pred, topk, indiv_topk)

    else:
        raise ValueError("bow strategy is not defined")

        
# [CLS]  [M]  w2  w3  [SEP]        
# [CLS]  w1  [M]  w3  [SEP]        
# [CLS]  w1 w2  [M]  [SEP]        
# Given the probability of [M] for each mask-prediction case

# sum all the probability and do topk to get bag of words        
def get_simple_sum_bow(n_words, pred, topk):
    bows = torch.zeros(n_words)
    for i in range(pred.shape[0]):
        prob = pred[i][i+1]
        bows += prob
    _, indices = torch.topk(bows, topk)
    
    return indices

# do topk first for each probability distribution
# sum them up and do topk again to get bag of words
def get_indiv_topk_bow(n_words, pred, topk, indiv_topk):
    # todo: try to improve efficiency with matrix calculation
    probs, indiv_indices = torch.topk(pred, indiv_topk)
    bows = torch.zeros(n_words)
    for i in range(indiv_indices.shape[0]):
        prob, indices = probs[i][i+1], indiv_indices[i][i+1]
        res = torch.zeros(n_words)
        res = res.scatter(0, indices, prob)
        bows += res
    _, indices = torch.topk(bows, topk)

    return indices


# do topk first for each probability distribution
# get the topk words for each mask words as bag of words
def get_indiv_neighbors_bow(pred, topk, indiv_topk):
    probs, indiv_indices = torch.topk(pred, indiv_topk)
    final_indices = []
    for i in range(indiv_indices.shape[0]):
        _, indices = probs[i][i+1], indiv_indices[i][i+1]
        final_indices.append(indices)

    return torch.cat(final_indices)


# In[ ]:




