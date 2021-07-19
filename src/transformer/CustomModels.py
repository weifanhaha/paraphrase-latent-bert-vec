#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from transformer.Models import get_pad_mask, get_subsequent_mask
from transformer.Models import PositionalEncoding, Encoder


# In[ ]:


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


# In[ ]:



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


# In[ ]:


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




# In[ ]:


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, bert_out, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
#         dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask, return_attns=True)
        bert_out = self.model.probnn(bert_out)
        dec_output, dec_slf_attn_list, dec_enc_attn_list = self.model.decoder(trg_seq, trg_mask, enc_output, bert_out, src_mask, return_attns=True)
        
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1), dec_enc_attn_list


    def _get_init_state(self, src_seq, src_mask, bert_out):
        beam_size = self.beam_size

        enc_output, *_ = self.model.encoder(src_seq, src_mask)
        dec_output, dec_enc_attn_list = self._model_decode(self.init_seq, enc_output, bert_out, src_mask)
        
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        bert_out = bert_out.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores, dec_enc_attn_list, bert_out


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores


    def translate_sentence(self, src_seq, bert_out, return_attn=False):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 
        
        with torch.no_grad():
        #     pred, enc_slf_attn_list, dec_slf_attn_list, dec_enc_attn_list = model(src_seq, trg_seq, True)
        #     pred_seq = translator.translate_sentence(src_seq)
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores, dec_enc_attn_list, bert_out = self._get_init_state(src_seq, src_mask, bert_out)

            output_dec_enc_attn = [[] for _ in range(len(dec_enc_attn_list))]
            for l in range(len(dec_enc_attn_list)):
                    output_dec_enc_attn[l].append(dec_enc_attn_list[l][0,:,-1,:])    
            ans_idx = 0   # default


            for step in range(2, max_seq_len):    # decode up to max length
#                 print('enc_out shape: ', enc_output.shape)
#                 print('bert_out shape: ', bert_out.shape)

                dec_output, dec_enc_attn_list = self._model_decode(gen_seq[:, :step], enc_output, bert_out, src_mask)

                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
                
                # append dec_enc_attn_list
                for l in range(len(dec_enc_attn_list)):
                    output_dec_enc_attn[l].append(dec_enc_attn_list[l][0,:,-1,:])

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break


        if return_attn:
            for l in range(len(output_dec_enc_attn)):
                output_dec_enc_attn[l] = torch.stack(output_dec_enc_attn[l], dim=1)
            return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist(), output_dec_enc_attn
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()




