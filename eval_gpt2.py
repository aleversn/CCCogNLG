# %%
from CC.trainer import *
from transformers import GPT2Tokenizer

# %%
tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2')
tokenizer.pad_token = '<|pad|>'
tokenizer.sep_token = '<|sep|>'
trainer = Trainer(tokenizer, model_dir='model/gpt2', dataset_name='cnn', padding_length=200, batch_size=16)

# %%
# Common Training
trainer.eval(0, 0, gpu=[0, 1, 2, 3], resume_path='model/cnn/gpt2/epoch_10.pth', eval_mode='test', save_pred_dir='log')