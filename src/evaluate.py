#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
import rouge
from argparse import ArgumentParser
import yaml
import re

# to do : evaluate with ibleu


# In[ ]:


parser = ArgumentParser()
parser.add_argument("--config_path", dest="config_path",
                    default='../configs/dpng_transformer_bert_tokenizer_bow_indivtopk.yaml')

args = parser.parse_args()
config_path = args.config_path
print("config_path:", config_path)


# In[69]:


# config_path = '../configs/base_transformer.yaml'
# config_path = '../configs/dpng_transformer.yaml'
# config_path = '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk.yaml'
# config_path =  '../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_onlybow.yaml'


# In[70]:



with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

prediction_path = config['test_output_file']
eval_log_path = prediction_path.replace('outputs', 'scores')

train_size = config['train_size']
val_size = config['val_size']
test_size = config['test_size']


# In[71]:


reference_path = '../data/quora_train.txt'


# In[72]:


##### Arguments #####
# Arguments of DNPG

# train_size = 100000
# val_size = 4000
# test_size = 20000

# prediction_path = '../outputs/test_DNPG_transformer_out.txt'
# eval_log_path = '../scores/test_DNPG_transformer_out.txt'

# prediction_path = '../outputs/test_DNPG_transformer_bert_tokenizer_out.txt'
# eval_log_path = '../scores/test_DNPG_transformer_bert_tokenizer_out.txt'

# prediction_path = '../outputs/test_DNPG_transformer_bert_tokenizer_bow_out.txt'
# eval_log_path = '../scores/test_DNPG_transformer_bert_tokenizer_bow_out.txt'

# prediction_path = '../outputs/test_DNPG_transformer_bert_tokenizer_bow_indivtopk_out.txt'
# eval_log_path = '../scores/test_DNPG_transformer_bert_tokenizer_bow_indivtop_out.txt'

###################


# In[73]:


# read reference sentence and prediction sentence
reference_text = open(reference_path, 'r').readlines()
reference_text = reference_text[train_size+val_size:train_size+val_size+test_size]
reference_text = [text.strip().split('\t')[1] for text in reference_text]

# normalize reference corpus , eg: seperate question mark , remove '(' ,')' etc
reference_text = [re.sub(r"([.!?])", r" \1", seq) for seq in reference_text]
# reference_text = [re.sub(r"[^a-zA-Z.!?]+", r" ", seq) for seq in reference_text]
reference_corpus = [[text.split()] for text in reference_text]


# In[74]:


prediction_text = open(prediction_path, 'r').readlines()
prediction_text = [text.replace('Predict: ', '').strip() for text in prediction_text if 'Predict: ' in text]
prediction_corpus = [text.split() for text in prediction_text]


# In[75]:


print(len(reference_text))
print(len(prediction_text))


# In[76]:


# i = 15
# print(reference_corpus[i])
# print(prediction_corpus[i])
# # sentence_bleu(reference_corpus[i], prediction_corpus[i], weights=(0.33, 0.33, 0.34, 0))
# sentence_bleu(reference_corpus[i], prediction_corpus[i], weights=(0.25, 0.25, 0.25, 0.25))

# # r = [['Why', 'is', 'chemistry', 'so', 'boring', '?']]
# # p = ['Why', 'is', 'chemistry', 'so', 'boring', '?']
# # sentence_bleu(r, p, weights=(0.25, 0.25, 0.25, 0.25))


# In[77]:


# print(reference_corpus[:6])
# print(prediction_corpus[:6])


# In[78]:


print("[Info] Calculating BLEU 1...")
bleu1 = corpus_bleu(reference_corpus, prediction_corpus, weights=(1, 0, 0, 0))
print("[Info] Calculating BLEU 2...")
bleu2 = corpus_bleu(reference_corpus, prediction_corpus, weights=(0.5, 0.5, 0, 0))
print("[Info] Calculating BLEU 3...")
bleu3 = corpus_bleu(reference_corpus, prediction_corpus, weights=(0.33, 0.33, 0.34, 0))
print("[Info] Calculating BLEU 4...")
bleu4 = corpus_bleu(reference_corpus, prediction_corpus, weights=(0.25, 0.25, 0.25, 0.25))
print("[Info] Done")


# In[79]:


print("[Info] BLEU1 Score: {}".format(bleu1))
print("[Info] BLEU2 Score: {}".format(bleu2))
print("[Info] BLEU3 Score: {}".format(bleu3))
print("[Info] BLEU4 Score: {}".format(bleu4))


# In[66]:


rouge_evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
rouge_scores = rouge_evaluator.get_scores(prediction_text, reference_text)


# In[67]:


print("[Info] Rouge 1 score: {}".format(rouge_scores["rouge-1"]["f"]))
print("[Info] Rouge 2 score: {}".format(rouge_scores["rouge-2"]["f"]))
print("[Info] Rouge l score: {}".format(rouge_scores["rouge-l"]["f"]))


# In[68]:


# save to file
f = open(eval_log_path, 'w')
f.write("[Info] BLEU1 Score: {}\n".format(bleu1))
f.write("[Info] BLEU2 Score: {}\n".format(bleu2))
f.write("[Info] BLEU3 Score: {}\n".format(bleu3))
f.write("[Info] BLEU4 Score: {}\n".format(bleu4))

f.write("\n\n[Info] Rouge 1 score: {}\n".format(rouge_scores["rouge-1"]["f"]))
f.write("[Info] Rouge 2 score: {}\n".format(rouge_scores["rouge-2"]["f"]))
f.write("[Info] Rouge l score: {}\n".format(rouge_scores["rouge-l"]["f"]))

f.close()


# In[ ]:




