# %%
from transformers import ProphetNetTokenizer, ProphetNetConfig, ProphetNetModel

# %%
tokenizer = ProphetNetTokenizer.from_pretrained('model/prophetnet-large-uncased')
model = ProphetNetModel.from_pretrained('model/prophetnet-large-uncased')
input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

last_hidden_states = outputs.last_hidden_state  # main stream hidden states
last_hidden_states_ngram = outputs.last_hidden_state_ngram  # predict hidden states

# %%
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration

tokenizer = ProphetNetTokenizer.from_pretrained('model/prophetnet-large-uncased')
model = ProphetNetForConditionalGeneration.from_pretrained('model/prophetnet-large-uncased')

input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

logits_next_token = outputs.logits  # logits to predict next token as usual
logits_ngram_next_tokens = outputs.logits_ngram  # logits to predict 2nd, 3rd, ... next tokens


# %%
import torch
import torch.nn.functional as F

logits = outputs.logits
lprobs = F.log_softmax(
    logits.view(-1, logits.size(-1)),
    dim=-1,
    dtype=torch.float32,
)

loss = F.nll_loss(
    lprobs,
    decoder_input_ids.view(-1),
    reduction='sum',
    ignore_index=102,
)

pred = logits.max(-1)[1]

# %%
loss.backward()

# %%
from CC.evals.bleu.bleu import Bleu

res = {'1': ['who did the virgin mary appear to in 1858 ?']}
gts = {'1': ['to whom did the virgin mary allegedly appear in 1858 in lou ##rdes france ?']}
scorer = Bleu()
scorer.compute_score(gts, res)

# %%
from CC.trainer import *
from transformers import ProphetNetTokenizer, ProphetNetConfig, ProphetNetModel

# %%
tokenizer = ProphetNetTokenizer.from_pretrained('model/prophetnet-large-uncased')
trainer = Trainer(tokenizer, model_dir='model/prophetnet-large-uncased', dataset_name='qg', padding_length=100, batch_size=16)

# %%
# Common Training
trainer.train(lr=1e-5, gpu=[0, 1, 2, 3], resume_path='model/qg/prophetnet/epoch_1.pth', is_eval=True, eval_mode='test')

# %%
trainer.eval(0, 0, gpu=[0, 1, 2, 3], resume_path='model/qg/prophetnet/epoch_1.pth', eval_mode='test', save_pred_dir='log')

# %%
with open('datasets/qg_data/prophetnet_tokenized/test.src', mode='r') as f:
    src_list = f.read().split('\n')
    src_list = src_list[:-1]
with open('datasets/qg_data/prophetnet_tokenized/test.tgt', mode='r') as f:
    gold_list = f.read().split('\n')
    gold_list = gold_list[:-1]
with open('log/qg/prophetnet/predict.csv', mode='r') as f:
    pred_list = f.read().split('\n')
    pred_list = pred_list[:-1]
from CC.evals.bleu.bleu import Bleu
from CC.evals.rouge.rouge import Rouge
from collections import defaultdict
res = defaultdict(lambda: [])
gts = defaultdict(lambda: [])
for idx, _ in enumerate(src_list):
    res[_] = [pred_list[idx]]
    gts[_].append(gold_list[idx])
bleu = Bleu(4)
rouge = Rouge()
bleu_score, _ = bleu.compute_score(gts, res)
rouge_score, _ = rouge.compute_score(gts, res)
print(bleu_score, rouge_score)

# %%
# import pickle

# with open('log/qg/prophetnet/cache/cache.res', mode='rb') as f:
#     res = pickle.load(f)
# with open('log/qg/prophetnet/cache/cache.gts', mode='rb') as f:
#     gts = pickle.load(f)
# bleu = Bleu(4)
# rouge = Rouge()
# bleu_score, _ = bleu.compute_score(gts, res)
# rouge_score, _ = rouge.compute_score(gts, res)
# print(bleu_score, rouge_score)

# %%
from CC.predictor import *
from transformers import ProphetNetTokenizer, ProphetNetConfig, ProphetNetModel

# %%
tokenizer = ProphetNetTokenizer.from_pretrained('model/prophetnet-large-uncased')
predictor = Predictor(tokenizer, model_dir='model/prophetnet-large-uncased', resume_path='model/qg/prophetnet/epoch_5.pth', padding_length=128)

# %%
predictor('saint bern ##ade ##tte so ##ub ##iro ##us [SEP] architectural ##ly , the school has a catholic character . atop the main building \' s gold dome is a golden statue of the virgin mary . immediately in front of the main building and facing it , is a copper statue of christ with arms up ##rai ##sed with the legend \" ve ##ni ##te ad me om ##nes \" . next to the main building is the basilica of the sacred heart . immediately behind the basilica is the gr ##otto , a marian place of prayer and reflection . it is a replica of the gr ##otto at lou ##rdes , france where the virgin mary reputed ##ly appeared to saint bern ##ade ##tte so ##ub ##iro ##us in 1858 . at the end of the main drive ( and in a direct line that connects through 3 statues and the gold dome ) , is a simple , modern stone statue of mary .', 'to whom did the virgin mary allegedly appear in 1858 in lou ##rdes france ?')

# %%
import os
from CC.trainer import *
from transformers import GPT2Tokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# %%
tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = tokenizer.eos_token
trainer = Trainer(tokenizer, model_dir='model/gpt2', dataset_name='entdesc', padding_length=512, batch_size=4, batch_size_eval=1)

# %%
# Common Training
trainer.train(num_epochs=30, lr=5e-5, gpu=[0, 1], resume_path='./model/entdesc/gpt2/epoch_5.pth', eval_mode='test', is_eval=False)

# %%
trainer.eval(0, 0, gpu=[0, 1, 2, 3], resume_path='model/entdesc/gpt2/epoch_23.pth', eval_mode='test', save_pred_dir='log/entdesc/gpt2_23_pure')

# %%
trainer.eval_without_hint(0, 0, gpu=[0], resume_path='model/entdesc/gpt2/epoch_5.pth', eval_mode='test', save_pred_dir='log')

# %%
import os
import sys
from CC.predictor import *
from transformers import GPT2Tokenizer
from tqdm import tqdm

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = tokenizer.eos_token
pred = Predictor(tokenizer, model_dir='model/gpt2', padding_length=512, resume_path='./model/entdesc/gpt2/epoch_23.pth', gpu=[0, 1, 2])

# %%
context = ('Hubbard Landing Seaplane Base', 'seaplane', 'Baldwin County', 'Alabama', 'United States of America', 'nautical mile', 'mile', 'kilometre', 'central business district', 'Stockton')
pred_iter = tqdm(pred.predict_continous(*context))
for loss, logits, Y in pred_iter:
    pred_iter.set_description('⭐Preding⭐')
    pred_iter.set_postfix(content=tokenizer.decode(Y[0]))

print('\n⭐⭐⭐⭐⭐⭐⭐\n{}\n⭐⭐⭐⭐⭐⭐⭐'.format(tokenizer.decode(Y[0], skip_special_tokens=True)))

# %%
from nltk.corpus import wordnet
word = "Chinese"
synonyms = []

for syn in wordnet.synsets(word):
    for lm in syn.lemmas():
        synonyms.append(lm.name())
print (set(synonyms))

# %%
