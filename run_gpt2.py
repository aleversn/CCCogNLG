# %%
from CC.trainer import *
from transformers import GPT2Tokenizer

# %%
tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2')
tokenizer.pad_token = '<|pad|>'
tokenizer.sep_token = '<|sep|>'
trainer = Trainer(tokenizer, model_dir='model/gpt2', dataset_name='entdesc_triples', padding_length=1024, batch_size=2)

# %%
# Common Training
trainer.train(num_epochs=30, lr=5e-5, gpu=[0], resume_path='./model/entdesc/gpt2/epoch_23.pth', eval_mode='test', is_eval=False)

# %%
