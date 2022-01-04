# %%
from CC.CogNLG import *
from transformers import GPT2Tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# %%
tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = tokenizer.eos_token
trainer = Trainer(tokenizer, model_dir='model/gpt2', dataset_name='entdesc_cognlg', padding_length=512)

# %%
# Common Training
trainer.train(num_epochs=30, lr1=5e-5, gpu=[0], resume_path_sys1='./model/entdesc_cognlg/sys1/epoch_17.pth',  resume_path_sys2='./model/entdesc_cognlg/sys2/epoch_17.pth', entities_file_name='./datasets/ENT-DESC/train_mass_entities.txt', hidden_states_dir='/home/lpc/sdata/Stellar/v6/train_hidden', best_nodes_file_name='./datasets/ENT-DESC/best_nodes/train_5.txt', train_mode='both')

# %%
trainer.eval(0, 0, gpu=[0], resume_path_sys1='./model/entdesc_cognlg_sys1/sys1/epoch_13.pth', resume_path_sys2='./model/entdesc_cognlg/sys2/epoch_13.pth', entities_file_name='./datasets/ENT-DESC/test_mass_entities.txt', hidden_states_dir='/home/lpc/sdata/Stellar/v6/test_hidden', best_nodes_file_name='./datasets/ENT-DESC/best_nodes/test_6.txt', eval_mode='test', save_pred_dir='./log/entdesc_cognlg/gpt2_13')

# %%
from CC.process import *
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# %%
# 保存节点树
save_data_entities('./datasets/ENT-DESC/train-entity.txt', './datasets/ENT-DESC/train-triples.txt', './datasets/ENT-DESC/triple_types.csv', './datasets/ENT-DESC/train_mass_entities.txt')

# %%
# 保存节点隐藏状态
save_entities_hidden_state('./datasets/ENT-DESC/dev_mass_entities.txt', '/home/lpc/sdata/Stellar/v6/dev_hidden/')

# %%
save_tgt_ners('./datasets/ENT-DESC/test_surface.pp.txt', './datasets/ENT-DESC/tgt_ners/test.json')

# %%
save_triple_type_template(['./datasets/ENT-DESC/train-triples.txt', './datasets/ENT-DESC/dev-triples.txt', './datasets/ENT-DESC/test-triples.txt'], './datasets/ENT-DESC/triple_types.csv')

# %%
