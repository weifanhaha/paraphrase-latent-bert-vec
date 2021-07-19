config_path='../configs/dpng_transformer_bert_tokenizer_with_classifier.yaml'

# lambda = 0.1
cp ../models/tune/DNPG_base_transformer_bert_tokenizer_with_classifier_lambda0.1_batch50_warmup30000.pth ../models/DNPG_base_transformer_bert_tokenizer_with_classifier.pth 
python3 predict_bert.py --config_path $config_path 
python3 evaluate.py --config_path $config_path

mv ../outputs/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../outputs/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda0.1.txt
mv ../scores/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../scores/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda0.1.txt

# lambda = 1
cp ../models/tune/DNPG_base_transformer_bert_tokenizer_with_classifier_lambda1_batch50_warmup30000.pth ../models/DNPG_base_transformer_bert_tokenizer_with_classifier.pth 
python3 predict_bert.py --config_path $config_path 
python3 evaluate.py --config_path $config_path

mv ../outputs/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../outputs/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda1.txt
mv ../scores/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../scores/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda1.txt

# lambda = 10
cp ../models/tune/DNPG_base_transformer_bert_tokenizer_with_classifier_lambda10_batch50_warmup30000.pth ../models/DNPG_base_transformer_bert_tokenizer_with_classifier.pth 
python3 predict_bert.py --config_path $config_path 
python3 evaluate.py --config_path $config_path

mv ../outputs/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../outputs/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda10.txt
mv ../scores/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../scores/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda10.txt

# lambda = 100
cp ../models/tune/DNPG_base_transformer_bert_tokenizer_with_classifier_lambda100_batch50_warmup30000.pth ../models/DNPG_base_transformer_bert_tokenizer_with_classifier.pth 
python3 predict_bert.py --config_path $config_path 
python3 evaluate.py --config_path $config_path

mv ../outputs/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../outputs/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda100.txt
mv ../scores/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../scores/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda100.txt

# lambda = 1000
cp ../models/tune/DNPG_base_transformer_bert_tokenizer_with_classifier_lambda1000_batch50_warmup30000.pth ../models/DNPG_base_transformer_bert_tokenizer_with_classifier.pth 
python3 predict_bert.py --config_path $config_path 
python3 evaluate.py --config_path $config_path

mv ../outputs/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../outputs/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda1000.txt
mv ../scores/test_DNPG_transformer_bert_tokenizer_with_classifier_out.txt ../scores/tune/test_DNPG_transformer_bert_tokenizer_with_classifier_out_lambda1000.txt

