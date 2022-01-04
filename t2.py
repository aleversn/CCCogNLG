# %%
from CC.CogNLG import *
from transformers import GPT2Tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# %%
tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = tokenizer.eos_token
trainer = Trainer(tokenizer, model_dir='model/gpt2', dataset_name='entdesc_cognlg', padding_length=512)

# %%
trainer.eval(0, 0, gpu=[0], resume_path_sys1='./model/entdesc_cognlg/sys1/epoch_19.pth', resume_path_sys2='./model/entdesc_cognlg/sys2/epoch_19.pth', entities_file_name='./datasets/ENT-DESC/test_mass_entities.txt', hidden_states_dir='/home/lpc/sdata/Stellar/v6/test_hidden', best_nodes_file_name='./datasets/ENT-DESC/best_nodes/test_6.txt', eval_mode='test', save_pred_dir='./log/entdesc_cognlg/gpt2_19')

# %%
# import os
# from CC.trainer import *
# from transformers import GPT2Tokenizer
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# # %%
# tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2')
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.sep_token = tokenizer.eos_token
# trainer = Trainer(tokenizer, model_dir='model/gpt2', dataset_name='entdesc', padding_length=512, batch_size=4, batch_size_eval=1)

# # %%
# trainer.eval_without_hint(0, 0, gpu=[0], resume_path='model/entdesc/gpt2/epoch_12.pth', eval_mode='test', save_pred_dir='./log/entdesc/gpt2_12')

# %%
from CC.analysis import Analysis

bleu_scores = Analysis.compute_bleu_from_txt('./log/entdesc_cognlg/gpt2_18/predict_gold.csv')
rouge_scores = Analysis.compute_rouge_from_txt('./log/entdesc_cognlg/gpt2_18/predict_gold.csv')
nltk_bleu_scores = Analysis.compute_nltk_bleu_from_txt('./log/entdesc_cognlg/gpt2_18/predict_gold.csv')

print(bleu_scores, rouge_scores, nltk_bleu_scores)

# %%
