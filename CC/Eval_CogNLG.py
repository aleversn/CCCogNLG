import os
import json
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from CC.ICCStandard import ITrainer
from CC.model import AutoModel
from CC.loader import AutoDataloader
from CC.models.gcn import GCN
from CC.analysis import Analysis
from tqdm import tqdm
from torch.autograd import Variable
from transformers import GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.multiprocessing import Pool
from CC.CogNLG import *

this_trainer = 1
MODEL_DIR = 'model/gpt2'
DATASET_NAME = 'entdesc_cognlg'
PADDDING_LENGTH = 512
resume_path_sys1 = './model/entdesc_cognlg/sys1/epoch_32.pth'
resume_path_sys2 = './model/entdesc_cognlg/sys2/epoch_32.pth'
entities_file_name = './datasets/ENT-DESC/test_mass_entities.txt'
hidden_states_file_name = './datasets/ENT-DESC/test_entities_hidden_states'
best_nodes_file_name = './datasets/ENT-DESC/best_nodes/test.txt'
eval_mode = 'test'
save_pred_dir = './log/entdesc_cognlg/gpt2_32'
GPU = [0, 1, 2, 3]
top_k=8
top_p=0.5

tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = tokenizer.eos_token
trainer = Trainer(tokenizer, model_dir=MODEL_DIR, dataset_name=DATASET_NAME, padding_length=PADDDING_LENGTH)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_loader = trainer.eval_loader if eval_mode == 'dev' else trainer.test_loader
trainer.sep_token_ids = trainer.cuda(trainer.sep_token_ids)

# if torch.cuda.is_available() and trainer.model_cuda == False:
#     trainer.model = torch.nn.DataParallel(trainer.model, device_ids=GPU).cuda()
#     trainer.model.to(device)
#     trainer.gcn = torch.nn.DataParallel(trainer.gcn, device_ids=GPU).cuda()
#     trainer.gcn.to(device)
#     trainer.model_cuda = True
#     trainer.top_k = top_k
#     trainer.top_p = top_p

# if not resume_path_sys1 == False:
#     print('Accessing System 1 Resume PATH: {} ...\n'.format(resume_path_sys1))
#     model_dict = torch.load(resume_path_sys1).module.state_dict()
#     trainer.model.module.load_state_dict(model_dict)
#     trainer.model.to(device)

# if not resume_path_sys2 == False:
#     print('Accessing System 2 Resume PATH: {} ...\n'.format(resume_path_sys2))
#     model_dict = torch.load(resume_path_sys2).module.state_dict()
#     trainer.gcn.module.load_state_dict(model_dict)
#     trainer.gcn.to(device)

if torch.cuda.is_available() and trainer.model_cuda == False:
    trainer.model.to(device)
    trainer.gcn.to(device)
    trainer.model_cuda = True
    trainer.top_k = top_k
    trainer.top_p = top_p

if not resume_path_sys1 == False:
    print('Accessing System 1 Resume PATH: {} ...\n'.format(resume_path_sys1))
    trainer.model = torch.load(resume_path_sys1)
    trainer.model.to(device)

if not resume_path_sys2 == False:
    print('Accessing System 2 Resume PATH: {} ...\n'.format(resume_path_sys2))
    trainer.gcn = torch.load(resume_path_sys2)
    trainer.gcn.to(device)

if entities_file_name is not None:
    print('Accessing Entities From PATH: {} ... \n'.format(entities_file_name))
    trainer.e_list = data_loader.dataset.load_entities_from_file(entities_file_name)

if hidden_states_file_name is not None:
    print('Accessing Hidden States From PATH: {} ... \n'.format(hidden_states_file_name))
    with open(hidden_states_file_name, mode='rb') as f:
        trainer.x_list = pickle.load(f)
else:
        trainer.x_list = None

if best_nodes_file_name is not None:
    print('Accessing Hidden States From PATH: {} ... \n'.format(best_nodes_file_name))
    trainer.n_list = data_loader.dataset.load_best_nodes_from_file(best_nodes_file_name)

trainer.model.eval()
trainer.gcn.eval()

def eval_multiprocessing():
    eval_iter = data_loader
    eval_loss = []
    SRC = []
    TARGET = []
    PREDICTION = []
    ctx = torch.multiprocessing.get_context("spawn")
    eval_count = 0
    eval_iter = data_loader
    with ctx.Pool(2) as pool:
        for result in tqdm(pool.imap(run, eval_iter)):
            it, loss, pred_input = result
        
            eval_loss.append(loss.sum().data.item())
            
            cur_SRC = it['src_input_ids'].tolist()
            cur_TARGET = it['target_input_ids'].tolist()
            cur_PREDICTION = pred_input.unsqueeze(0).tolist()
            
            SRC += cur_SRC
            TARGET += cur_TARGET
            PREDICTION += cur_PREDICTION
            
            eval_count += 1
    
    print('Compuing Score ...')
    scores = trainer.analysis.Evaluation(SRC, TARGET, PREDICTION)
    print('\n', scores)

    if save_pred_dir != False:
        trainer.analysis.save_xy(TARGET, PREDICTION, save_pred_dir)
    
    return scores, np.mean(eval_loss)

def run(it):
    with torch.no_grad():
        for key in it.keys():
            it[key] = trainer.cuda(it[key])
        
        it['output_hidden_states'] = True
        it['output_attentions'] = True
        
        pred_input = torch.tensor([]).int()
        pred_input = trainer.cuda(pred_input)
        it['input_ids'] = it['src_input_ids']
        it['attention_mask'] = it['input_ids'].gt(0).int()
        semantic = trainer.model(**it).hidden_states[-1][:, -1, :]

        idx = it['idx'].tolist()[0]
        if entities_file_name is not None:
            e2i, i2e = trainer.e_list[idx]
        else:
            entities = data_loader.dataset.get_entities(idx)
            e2i, i2e = data_loader.dataset.load_entities(entities, hops=2)
        A = data_loader.dataset.construct_adjacent(e2i, i2e, trainer.config['n_embd'])
        A = A.unsqueeze(0)

        while len(pred_input) < 200:
            H = torch.zeros(semantic.size(1), semantic.size(1))
            if trainer.x_list is not None:
                h = trainer.x_list[idx]
                for key in h:
                    H[int(key)] = h[key]
            H = trainer.cuda(H)
            H = (H + semantic) / 2

            gcn_output = trainer.gcn(A, H.unsqueeze(0))
            l = len(e2i)
            gcn_output[0][l:] = -1e10
            gcn_pred = gcn_output.max(-1)[1]
            best_node = gcn_pred[0].tolist()

            bn = i2e[str(best_node)]
            best_node_context = trainer.tokenizer(bn['name'] + trainer.tokenizer.sep_token + bn['context'], max_length=trainer.padding_length, truncation=True)['input_ids']
            best_node_context = torch.tensor(best_node_context)
            best_node_context = trainer.cuda(best_node_context)

            input_ids = torch.cat([it['src_input_ids'], trainer.sep_token_ids.unsqueeze(0), best_node_context.unsqueeze(0), trainer.sep_token_ids.unsqueeze(0), pred_input.unsqueeze(0)], dim=-1)
            attention_mask = input_ids.gt(0).int()
            it['input_ids'] = input_ids
            it['attention_mask'] = attention_mask
            
            outputs = trainer.model(**it)
            gpt_loss = outputs.loss
            pred = outputs.logits

            hidden_states = outputs.hidden_states[-1]
            semantic = hidden_states[:, -1, :]

            loss = gpt_loss
            next_token_logits = pred[:, -1]
            filtered_logits = trainer.top_k_top_p_filtering(
                next_token_logits[0], top_k=trainer.top_k, top_p=trainer.top_p
            )
            next_token = torch.multinomial(
                torch.softmax(filtered_logits, dim=-1), num_samples=1
            )
            pred_input = torch.cat([pred_input, next_token], dim=0)
        
        return it, loss, pred_input