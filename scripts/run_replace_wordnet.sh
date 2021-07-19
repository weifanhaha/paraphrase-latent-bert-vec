#!/bin/bash

# config_path='../configs/base_transformer.yaml'
# config_path='../configs/dpng_transformer.yaml'
# config_path='../configs/dpng_transformer_bert_tokenizer.yaml'
# config_path='../configs/dpng_transformer_bert_tokenizer_bow.yaml'
# config_path='../configs/dpng_transformer_bert_tokenizer_bow_indivtopk.yaml'
# config_path='../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_replace.yaml'
# config_path='../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_replace_nopreprocess.yaml'
# config_path='../configs/dpng_transformer_wordnet.yaml'
# config_path='../configs/dpng_transformer_wordnet_replace_nopreprocess.yaml'
# config_path='../configs/dpng_transformer_bert_tokenizer_bow_indivtopk_replace_nopreprocess_no_append_bow.yaml'
# config_path='../configs/dpng_transformer_bert_tokenizer_with_classifier.yaml'

cd ../src/

# Modify this to control to start from which stage
# 1: preprocess, 2: train, 3: predict, 4: evaluate
stage=1
preprocess=0
# # seed = 0, 777, 33333
# # base

# run successfully
config_path='../configs/dpng_transformer_wordnet_replace_nopreprocess_no_append_bow.yaml'

# fix random seed
seed=0

# preprocess config
preprocess=0

if [[ $config_path == *"bert"* ]]; then
    use_bert=1
else
    use_bert=0
fi

if [ $preprocess -gt 0 ]; then
    preprocessed_cmd='--preprocessed'
else
    preprocessed_cmd=''
fi

if [ $stage -le 1 ]; then
    echo "### Stage 1: Preprocess the bert mask predict data and dump to np file ###"
    if [ $preprocess -gt 0 ]; then
        python3 preprocess_bert_predict.py --config_path $config_path
    fi
    echo "#############################################################"
fi

if [ $stage -le 2 ]; then
    echo "### Stage 2: Train Seq2seq model ###"
    python3 train.py --config_path $config_path $preprocessed_cmd --seed $seed
    echo "#############################################################"
fi

# todo: modify the predict and evaluate files
if [ $stage -le 3 ]; then
    echo "### Stage 3: Predict Seq2seq model with test data ###"
    if [ $use_bert == 1 ]; then
        python3 predict_bert.py --config_path $config_path $preprocessed_cmd --seed $seed
    else
        python3 predict.py --config_path $config_path --seed $seed
    fi
    echo "#############################################################"
fi

echo "-------------- change seed--------------"
seed=777

if [ $stage -le 1 ]; then
    echo "### Stage 1: Preprocess the bert mask predict data and dump to np file ###"
    if [ $preprocess -gt 0 ]; then
        python3 preprocess_bert_predict.py --config_path $config_path
    fi
    echo "#############################################################"
fi

if [ $stage -le 2 ]; then
    echo "### Stage 2: Train Seq2seq model ###"
    python3 train.py --config_path $config_path $preprocessed_cmd --seed $seed
    echo "#############################################################"
fi

# todo: modify the predict and evaluate files
if [ $stage -le 3 ]; then
    echo "### Stage 3: Predict Seq2seq model with test data ###"
    if [ $use_bert == 1 ]; then
        python3 predict_bert.py --config_path $config_path $preprocessed_cmd --seed $seed
    else
        python3 predict.py --config_path $config_path --seed $seed
    fi
    echo "#############################################################"
fi

echo "-------------- change seed--------------"
seed=33333


if [ $stage -le 1 ]; then
    echo "### Stage 1: Preprocess the bert mask predict data and dump to np file ###"
    if [ $preprocess -gt 0 ]; then
        python3 preprocess_bert_predict.py --config_path $config_path
    fi
    echo "#############################################################"
fi

if [ $stage -le 2 ]; then
    echo "### Stage 2: Train Seq2seq model ###"
    python3 train.py --config_path $config_path $preprocessed_cmd --seed $seed
    echo "#############################################################"
fi

# todo: modify the predict and evaluate files
if [ $stage -le 3 ]; then
    echo "### Stage 3: Predict Seq2seq model with test data ###"
    if [ $use_bert == 1 ]; then
        python3 predict_bert.py --config_path $config_path $preprocessed_cmd --seed $seed
    else
        python3 predict.py --config_path $config_path --seed $seed
    fi
    echo "#############################################################"
fi







