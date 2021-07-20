# Transformer with Latent BERT Vector

This project is the implementation of thesis "Utilizing BERT to Explore Semantically Similar Words for Paraphrase Generation". We use seq2seq Transformer model to generate paraphrase and utilize the output distribution of BERT mask prediction as latent vectors in customized Transformer decoder attention block.

## Repo Structures
* analysis: some code to analyze the result and some plotted figures
* **configs**: config the model and dataset setting here
* data: the training data and the preprocessed files (.npy files are not added to git)
* logs: the training logs
* models: where the pytorch models stored, not added to git
* outputs: the output texts model predicts, there are some jupyter notebooks to calculate the metrics
* scores: the metrics scores of the generated paraphrases
* **scripts**: to run preprocess, train, and predict in one time
* **src**: the pytorch datasets, train and prediction implemetation


## How to run the code

### Install the requirements

```
pip install -r requirement.txt
```

### Run with scripts
For baseline model and baseline model with data augmentation, you can modify the config and run the scripts in the `scripts/` folder. The scripts will run preprocess, train and predict. 
```
cd scripts
./run_dnpg_wordnet_aug_uncased.sh
```

### Run with python
For baseline model and baseline model with data augmentation, you can run preprocess, train and predict on your own:
```
python3 preprocess_bert_predict.py --config_path $config_path
python3 train.py --config_path $config_path $preprocessed_cmd --seed $seed
python3 predict.py --config_path $config_path --seed $seed
```
Notice that if you use bert tokenizer(the config name should contain *bert*), the predict file is `predict_bert.py`:
```
python3 predict_bert.py --config_path $config_path $preprocessed_cmd --seed $seed
```

For the Transformer with latent BERT vector, you need to choose the config and run:
```
python3 train_transformer_with_latent_bert.py
python3 predict_transformer_with_latent_bert.py
```


## Todos and Future Work
1. Integrates the datasets if possible.
2. Add a script to train and predict Transformer with latent BERT vector
3. Accelerate the process the train Transformer with latent BERT vector by preprocess or other technique. (We have tried to preprocess and store all the mask prediction probability but still takes a lot of time to read the preprocess files and train the model)
4. Implement the future work write in the thesis.
    * Tune or learn alpha to get the better results.
    * Do experiment on MSCOCO or other dataset.

### Some notes
* Some experiments are not written in the thesis (such as BOW experiments) since the results are not good. You may feel free to remove the useless code.
* For now we do mask prediction in every epoch for Transformer with latent BERT vecotrs, you may try to figure out some way to accelerate the process.

