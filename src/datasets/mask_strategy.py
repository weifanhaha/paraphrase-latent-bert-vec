import torch

def simple_sim(n_words, pred, replaced_sentence=None):
    bows = torch.zeros(n_words)
    for i in range(pred.shape[0]):
        prob = pred[i][i+1]
        bows += prob
    _, indices = torch.topk(bows, self.topk)
    return indices, replaced_sentence

def topk_first(n_words, indiv_topk, pred, replaced_sentence=None)
    # todo: try to improve efficiency with matrix calculation
    probs, indiv_indices = torch.topk(pred, indiv_topk)
    bows = torch.zeros(n_words)
    for i in range(indiv_indices.shape[0]):
        prob, indices = probs[i][i+1], indiv_indices[i][i+1]
        res = torch.zeros(n_words)
        res = res.scatter(0, indices, prob)
        bows += res
    _, indices = torch.topk(bows, topk)
    
    return indices, replaced_sentence
        
def indiv_neighbors(n_words, indiv_topk, pred, replaced_sentence=None):
    probs, indiv_indices = torch.topk(pred, indiv_topk)
    final_indices = []
    for i in range(indiv_indices.shape[0]):
        _, indices = probs[i][i+1], indiv_indices[i][i+1]
        final_indices.append(indices)

    return torch.cat(final_indices), replaced_sentence