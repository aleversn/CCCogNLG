import os
import json
import torch
import random
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
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.multiprocessing import Pool

class Trainer(ITrainer):

    def __init__(self, tokenizer, model_dir, dataset_name, padding_length=128):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.padding_length = padding_length
        self.model_init(tokenizer, model_dir)
        self.dataloader_init(tokenizer, dataset_name, self.config['model_type'], padding_length)
        self.sep_token_ids = torch.tensor(self.tokenizer(self.tokenizer.sep_token)['input_ids'])
        self.analysis = Analysis(tokenizer)
    
    def model_init(self, tokenizer, model_dir):
        a = AutoModel(tokenizer, model_dir)
        print('AutoModel Choose Model: {}\n'.format(a.model_name))
        self.model_cuda = False
        self.config = a.config
        self.model = a()
        self.gcn = GCN(self.config['n_embd'])

    def dataloader_init(self, tokenizer, data_name, model_type, padding_length, batch_size=1, batch_size_eval=1):
        d = AutoDataloader(tokenizer, data_name, model_type, padding_length)
        self.train_loader, self.eval_loader, self.test_loader = d(batch_size, batch_size_eval)
    
    def __call__(self, resume_path=False, num_epochs=10, lr1=5e-5, lr2=1e-4, gpu=[0, 1, 2, 3], is_eval='train/eval', eval_mode='dev'):
        self.train(resume_path, num_epochs, lr1, lr2, gpu, is_eval, eval_mode)

    def train(self, resume_path_sys1=False, resume_path_sys2=False, entities_file_name=None, hidden_states_dir=None, best_nodes_file_name=None, num_epochs=10, lr1=5e-5, lr2=1e-4, gpu=[0, 1, 2, 3], train_mode='both', fp16=False, fp16_opt_level='O1'):
        '''
        train_mode: 'both' or 'sys2'.
        is_eval: decide whether to eval, True - both training and evaluating; False - only training.
        eval_mode: 'dev' or 'test'.
        fp16_opt_level: For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.gcn.to(device)
        
        optimizer = optim.AdamW([
            {'params': self.model.parameters(), 'lr': lr1},
            {'params': self.gcn.parameters(), 'lr': lr2}
        ],lr=lr1, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, 80000)
        node_loss = nn.BCELoss()

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=fp16_opt_level)
            self.gcn, optimizer = amp.initialize(self.gcn, optimizer, opt_level=fp16_opt_level)
        
        
        if torch.cuda.is_available() and self.model_cuda == False:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
            self.model.to(device)
            self.gcn = torch.nn.DataParallel(self.gcn, device_ids=gpu).cuda()
            self.gcn.to(device)
            self.model_cuda = True

        if not resume_path_sys1 == False:
            print('Accessing System 1 Resume PATH: {} ...\n'.format(resume_path_sys1))
            model_dict = torch.load(resume_path_sys1).module.state_dict()
            self.model.module.load_state_dict(model_dict)
            self.model.to(device)
        
        if not resume_path_sys2 == False:
            print('Accessing System 2 Resume PATH: {} ...\n'.format(resume_path_sys2))
            model_dict = torch.load(resume_path_sys2).module.state_dict()
            self.gcn.module.load_state_dict(model_dict)
            self.gcn.to(device)
        
        if entities_file_name is not None:
            print('Accessing Entities From PATH: {} ... \n'.format(entities_file_name))
            self.e_list = self.train_loader.dataset.load_entities_from_file(entities_file_name)
        
        if best_nodes_file_name is not None:
            print('Accessing Hidden States From PATH: {} ... \n'.format(best_nodes_file_name))
            self.n_list = self.train_loader.dataset.load_best_nodes_from_file(best_nodes_file_name)
        
        Epoch_loss_gpt2 = []
        Epoch_loss_gcn = []
        Epoch_node_pred_acc = []
        Epoch_node_pred_recall = []
        Epoch_node_pred_f1 = []

        Epoch_loss_eval = []
        Epoch_score_eval = []
        for epoch in range(num_epochs):
            train_count = 0
            train_loss_gpt2 = [0, 0]
            train_loss_gcn = [0, 0]
            train_node_pred_acc = [0, 0, 0] # TP, (TP + FP), TT
            train_iter = tqdm(self.eval_loader) if train_mode == 'eval' else tqdm(self.train_loader)
            if train_mode == 'sys1':
                self.model.train()
                self.gcn.eval()
            elif train_mode == 'sys2':
                self.model.eval()
                self.gcn.train()
            elif train_mode == 'none' or train_mode == 'eval':
                self.model.eval()
                self.gcn.eval()
            else:
                self.model.train()
                self.gcn.train()
            for it in train_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])
                self.sep_token_ids = self.cuda(self.sep_token_ids)
                
                it['output_hidden_states'] = True
                it['output_attentions'] = True
                
                end_idx = 0
                best_node = -1
                idx = it['idx'].tolist()[0]
                if entities_file_name is not None:
                    e2i, i2e = self.e_list[idx]
                else:
                    entities = self.train_loader.dataset.get_entities(idx)
                    e2i, i2e = self.train_loader.dataset.load_entities(entities, hops=2)
                A = self.train_loader.dataset.construct_adjacent(e2i, i2e, self.config['n_embd'])
                A = A.unsqueeze(0)
                # 构建节点隐藏表征 Matrix (hidden_size, hidden_size)
                cur_H = torch.zeros(self.config['n_embd'], self.config['n_embd'])
                if hidden_states_dir is not None:
                    with open(os.path.join(hidden_states_dir, '{}.hs'.format(idx)), mode='rb') as f:
                        h = pickle.load(f)
                        for key in h:
                            if int(key) < self.config['n_embd']:
                                cur_H[int(key)] = h[key]
                cur_H = self.cuda(cur_H)
                for gold_node_idx, gold_node in enumerate(self.n_list[idx]):
                    gold_node['ids'] = [int(i) for i in gold_node['ids']]
                    node_label = torch.zeros(1, self.config['n_embd'])
                    node_label[0][gold_node['ids']] = 1
                    node_label.float()
                    node_label = self.cuda(node_label)

                    extension = ''
                    for key in i2e:
                        bn = i2e[str(key)]
                        context = bn['context'] if bn['context'] != '' else bn['name']
                        extension += context if extension == '' else self.tokenizer.sep_token + context
                    best_node_context = self.tokenizer(extension, max_length=self.padding_length, truncation=True)['input_ids']
                    best_node_context = torch.tensor(best_node_context)
                    best_node_context = self.cuda(best_node_context)
                    
                    self.model.zero_grad()
                    self.gcn.zero_grad()
                    
                    input_ids = torch.cat([it['src_input_ids'], self.sep_token_ids.unsqueeze(0), best_node_context.unsqueeze(0), self.sep_token_ids.unsqueeze(0), it['target_input_ids']], dim=-1)
                    # end_idx = len(it['src_input_ids'][0]) + 1 + len(best_node_context) + 1 + gold_node['start']
                    end_idx = len(it['src_input_ids'][0]) + 1 + len(best_node_context) + 1
                    
                    it['input_ids'] = input_ids.long()
                    it['attention_mask'] = input_ids.gt(50256).int()
                    it['labels'] = input_ids.clone()
                    it['labels'][0][:end_idx - 1] = -100
                    outputs = self.model(**it)
                    gpt_loss = outputs.loss
                    pred = outputs.logits

                    hidden_states = outputs.hidden_states[-1]
                    # try:
                    #     print(hidden_states.shape, best_node, len(it['src_input_ids'][0]), len(best_node_context), len(it['target_input_ids'][0]), gold_node['start'], end_idx, input_ids.shape)
                    # except:
                    #     pass
                    try:
                        semantic = hidden_states[:, end_idx + gold_node['start'], :]
                    except:
                        semantic = hidden_states[:, -1, :]
                    semantic_expand = torch.repeat_interleave(semantic, repeats=semantic.size(1), dim=0)

                    # attentions = outputs.attentions
                    # _ = attentions[-1][0][:, :50, :50].tolist()
                    # for at in _:
                    #     x = np.array(at)
                    #     sns.heatmap(x, vmin=0, vmax=1, center=0.5)
                    #     plt.show()
                    # print(0 / 0)

                    gcn_output = self.gcn(A, cur_H.unsqueeze(0), semantic_expand.unsqueeze(0))
                    # mask掉不存在实体的信息
                    l = len(e2i)
                    # gcn_output[0][l:] = 0
                    gcn_pred = gcn_output.max(-1)[1]
                    gcn_loss = node_loss(gcn_output[:, :l, 1], node_label[:, :l])
                    # 选择使用预测的最佳节点或是使用训练集预提供的Gold Node
                    # best_node = gcn_pred[0].tolist()
                    best_node = gold_node['ids']

                    c_1 = 0
                    t_1 = 0
                    g_1 = 0
                    for i in range(0, l):
                        if gcn_pred[0][i] == 1:
                            t_1 += 1
                            if node_label[0][i].int() == 1:
                                c_1 += 1
                        if node_label[0][i].int() == 1:
                            g_1 += 1
                    train_node_pred_acc[0] += c_1
                    train_node_pred_acc[1] += t_1
                    train_node_pred_acc[2] += g_1
                    P = train_node_pred_acc[0] / train_node_pred_acc[1] if train_node_pred_acc[1] != 0 else 1
                    R = train_node_pred_acc[0] / train_node_pred_acc[2] if train_node_pred_acc[2] != 0 else 1
                    F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0

                    if train_mode != 'sys2':
                        loss = gpt_loss.mean() + gcn_loss.mean()
                    else:
                        loss = gcn_loss.mean()
                    
                    if train_mode == 'sys1':
                        loss = gpt_loss.mean()
                    elif train_mode == 'sys2':
                        loss = gcn_loss.mean()
                    elif train_mode == 'none' or train_mode == 'eval':
                        loss = gpt_loss.mean() + gcn_loss.mean()
                    else:
                        loss = gpt_loss.mean() + gcn_loss.mean()

                    optimizer.zero_grad()
                    if fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule


                    train_loss_gpt2[0] += (gpt_loss.data.item())
                    train_loss_gpt2[1] += 1
                    train_loss_gcn[0] += (gcn_loss.data.item())
                    train_loss_gcn[1] += 1
                    
                    train_count += 1

                    train_iter.set_description('Train: {}/{}'.format(epoch + 1, num_epochs))
                    train_iter.set_postfix(loss_gpt2=train_loss_gpt2[0]/train_loss_gpt2[1], loss_gcn=train_loss_gcn[0]/train_loss_gcn[1], node_idx=gold_node_idx, node_p=P, node_r=R, node_f1=F1)
                    break
            
            Epoch_loss_gpt2.append(train_loss_gpt2[0]/train_loss_gpt2[1])
            Epoch_loss_gcn.append(train_loss_gcn[0]/train_loss_gcn[1])
            Epoch_node_pred_acc.append(P)
            Epoch_node_pred_recall.append(R)
            Epoch_node_pred_f1.append(F1)
            
            _dir = './log/{}/{}'.format(self.dataset_name, self.config["model_type"])
            Analysis.save_same_row_list(_dir, 'eval_log' if train_mode == 'eval' else 'train_log', gpt_loss=Epoch_loss_gpt2, gcn_loss=Epoch_loss_gcn, node_pred_acc=Epoch_node_pred_acc, node_pred_recall=Epoch_node_pred_recall, node_pred_f1=Epoch_node_pred_f1)
            if resume_path_sys1 == False:
                self.save_model(epoch, 0)
            else:
                self.save_model(epoch, int(resume_path_sys1.split('/')[-1].split('_')[1].split('.')[0]))

    def save_model(self, epoch, save_offset=0):
        _dir_sys1 = './model/{}/sys1'.format(self.dataset_name)
        _dir_sys2 = './model/{}/sys2'.format(self.dataset_name)
        if not os.path.isdir(_dir_sys1):
            os.makedirs(_dir_sys1)
        if not os.path.isdir(_dir_sys2):
            os.makedirs(_dir_sys2)
        torch.save(self.model, '{}/epoch_{}.pth'.format(_dir_sys1, epoch + 1 + save_offset))
        torch.save(self.gcn, '{}/epoch_{}.pth'.format(_dir_sys2, epoch + 1 + save_offset))
    
    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert (
            logits.dim() == 1
        )  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            # torch.topk返回值和索引
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # 按维度里元素顺序累加并按顺序输出累加值
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            # 计算出累加概率大于p的索引
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            # 右移索引确保第一个token不会被1到
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 这种操作会筛选出sorted_indices_to_remove为1的值
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def eval(self, epoch, num_epochs, resume_path_sys1=False, resume_path_sys2=False, entities_file_name=None, hidden_states_dir=None, best_nodes_file_name=None, gpu=[0, 1, 2, 3], eval_mode='dev', save_pred_dir=False, top_k=8, top_p=0.5):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and self.model_cuda == False:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
            self.model.to(device)
            self.gcn = torch.nn.DataParallel(self.gcn, device_ids=gpu).cuda()
            self.gcn.to(device)
            self.model_cuda = True
        
        self.top_k = top_k
        self.top_p = top_p
        data_loader = self.eval_loader if eval_mode == 'dev' else self.test_loader

        if not resume_path_sys1 == False:
            print('Accessing System 1 Resume PATH: {} ...\n'.format(resume_path_sys1))
            model_dict = torch.load(resume_path_sys1).module.state_dict()
            self.model.module.load_state_dict(model_dict)
            self.model.to(device)
        
        if not resume_path_sys2 == False:
            print('Accessing System 2 Resume PATH: {} ...\n'.format(resume_path_sys2))
            model_dict = torch.load(resume_path_sys2).module.state_dict()
            self.gcn.module.load_state_dict(model_dict)
            self.gcn.to(device)
        
        if entities_file_name is not None:
            print('Accessing Entities From PATH: {} ... \n'.format(entities_file_name))
            self.e_list = data_loader.dataset.load_entities_from_file(entities_file_name)
        
        if best_nodes_file_name is not None:
            print('Accessing Hidden States From PATH: {} ... \n'.format(best_nodes_file_name))
            self.n_list = data_loader.dataset.load_best_nodes_from_file(best_nodes_file_name)

        with torch.no_grad():
            eval_loss = []
            SRC = []
            TARGET = []
            PREDICTION = []
            BESTNODES = []
            eval_count = 0
            self.model.eval()
            self.gcn.eval()
            eval_iter = tqdm(data_loader)
            for it in eval_iter:
                self.model.eval()
                for key in it.keys():
                    it[key] = self.cuda(it[key])
                
                self.sep_token_ids = self.cuda(self.sep_token_ids)
                
                it['output_hidden_states'] = True
                it['output_attentions'] = True
                
                pred_input = torch.tensor([]).int()
                pred_input = self.cuda(pred_input)
                pred_best_node_idx = []
                it['input_ids'] = it['src_input_ids']
                it['attention_mask'] = it['input_ids'].gt(50256).int()
                semantic = self.model(**it).hidden_states[-1][:, -1, :]

                idx = it['idx'].tolist()[0]
                if entities_file_name is not None:
                    e2i, i2e = self.e_list[idx]
                else:
                    entities = data_loader.dataset.get_entities(idx)
                    e2i, i2e = data_loader.dataset.load_entities(entities, hops=2)
                A = data_loader.dataset.construct_adjacent(e2i, i2e, self.config['n_embd'])
                A = A.unsqueeze(0)
                # 构建节点隐藏表征 Matrix (hidden_size, hidden_size)
                cur_H = torch.zeros(self.config['n_embd'], self.config['n_embd'])
                if hidden_states_dir is not None:
                    with open(os.path.join(hidden_states_dir, '{}.hs'.format(idx)), mode='rb') as f:
                        h = pickle.load(f)
                        for key in h:
                            if int(key) < self.config['n_embd']:
                                cur_H[int(key)] = h[key]
                cur_H = self.cuda(cur_H)

                while len(pred_input) < 100:
                    semantic_expand = torch.repeat_interleave(semantic, repeats=semantic.size(1), dim=0)

                    gcn_output = self.gcn(A, cur_H.unsqueeze(0), semantic_expand.unsqueeze(0))
                    l = len(e2i)
                    gcn_output[0][l:] = -1e10
                    gcn_pred = gcn_output.max(-1)[1]
                    best_node = []
                    for i in range(gcn_pred.shape[1]):
                        if gcn_pred[0][i].tolist() == 1:
                            best_node.append(i)
                    
                    extension = ''
                    for key in best_node:
                        bn = i2e[str(key)]
                        context = bn['context'] if bn['context'] != '' else bn['name']
                        extension += context if extension == '' else self.tokenizer.sep_token + context
                    # bn = i2e[str(random.randint(0, len(i2e) - 1))]
                    pred_best_node_idx.append(best_node)
                    best_node_context = self.tokenizer(extension, max_length=self.padding_length, truncation=True)['input_ids']
                    best_node_context = torch.tensor(best_node_context)
                    best_node_context = self.cuda(best_node_context)
                    

                    input_ids = torch.cat([it['src_input_ids'], self.sep_token_ids.unsqueeze(0), best_node_context.unsqueeze(0), self.sep_token_ids.unsqueeze(0), pred_input.unsqueeze(0)], dim=-1)
                    it['input_ids'] = input_ids.long()
                    it['attention_mask'] = input_ids.gt(50256).int()
                    
                    outputs = self.model(**it)
                    gpt_loss = outputs.loss
                    pred = outputs.logits

                    hidden_states = outputs.hidden_states[-1]
                    semantic = hidden_states[:, -1, :]

                    loss = gpt_loss
                    next_token_logits = pred[:, -1]
                    filtered_logits = self.top_k_top_p_filtering(
                        next_token_logits[0], top_k=self.top_k, top_p=self.top_p
                    )
                    next_token = torch.multinomial(
                        torch.softmax(filtered_logits, dim=-1), num_samples=1
                    )
                    pred_input = torch.cat([pred_input, next_token], dim=0)
                    eval_iter.set_postfix(length=pred_input.size(0))
                    

                eval_loss.append(loss.sum().data.item())
                
                cur_SRC = it['src_input_ids'].tolist()
                cur_TARGET = it['target_input_ids'].tolist()
                cur_PREDICTION = pred_input.unsqueeze(0).tolist()
                
                SRC += cur_SRC
                TARGET += cur_TARGET
                PREDICTION += cur_PREDICTION
                BESTNODES.append(pred_best_node_idx)
                
                eval_count += 1
                
                eval_iter.set_description('Eval: {}/{}'.format(epoch + 1, num_epochs))
            
            print('Compuing Score ...')
            scores = self.analysis.Evaluation(SRC, TARGET, PREDICTION)
            print('\n', scores)

            if save_pred_dir != False:
                self.analysis.save_xy_with_best_nodes(TARGET, PREDICTION, BESTNODES, save_pred_dir)
            
            return scores, np.mean(eval_loss)
        
    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX

def run(it):
    for key in it.keys():
        it[key] = it[key].cuda()
    return it