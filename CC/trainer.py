import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from CC.ICCStandard import ITrainer
from CC.model import AutoModel
from CC.loader import AutoDataloader
from CC.analysis import Analysis
from tqdm import tqdm
from torch.autograd import Variable
from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer(ITrainer):

    def __init__(self, tokenizer, model_dir, dataset_name, padding_length=128, batch_size=16, batch_size_eval=64):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.model_init(tokenizer, model_dir)
        self.padding_length = padding_length
        self.dataloader_init(tokenizer, dataset_name, self.config['model_type'], padding_length, batch_size, batch_size_eval)
        self.analysis = Analysis(tokenizer)
    
    def model_init(self, tokenizer, model_dir):
        a = AutoModel(tokenizer, model_dir)
        print('AutoModel Choose Model: {}\n'.format(a.model_name))
        self.model_cuda = False
        self.config = a.config
        self.model = a()

    def dataloader_init(self, tokenizer, data_name, model_type, padding_length, batch_size=16, batch_size_eval=64):
        d = AutoDataloader(tokenizer, data_name, model_type, padding_length)
        self.train_loader, self.eval_loader, self.test_loader = d(batch_size, batch_size_eval)
    
    def __call__(self, resume_path=False, num_epochs=10, lr=5e-5, gpu=[0, 1, 2, 3], is_eval='train/eval', eval_mode='dev'):
        self.train(resume_path, num_epochs, lr, gpu, is_eval, eval_mode)

    def train(self, resume_path=False, num_epochs=10, lr=5e-5, gpu=[0, 1, 2, 3], is_eval='train/eval', eval_mode='dev', fp16=False, fp16_opt_level='O1'):
        '''
        is_eval: decide whether to eval, True - both training and evaluating; False - only training.
        eval_mode: 'dev' or 'test'.
        fp16_opt_level: For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, 80000)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=fp16_opt_level)
        
        
        if torch.cuda.is_available() and self.model_cuda == False:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
            self.model_cuda = True
            self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(model_dict)
            self.model.to(device)
        
        Epoch_loss = []

        Epoch_loss_eval = []
        Epoch_score_eval = []
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = []
            train_iter = tqdm(self.train_loader)
            self.model.train()
            for it in train_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])
                self.model.zero_grad()
                
                outputs = self.model(**it)
                loss = outputs.loss
                pred = outputs.logits
                loss = loss.mean()

                optimizer.zero_grad()
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                train_loss.append(loss.data.item())
                
                train_count += 1

                train_iter.set_description('Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(train_loss=np.mean(train_loss))
            
            Epoch_loss.append(np.mean(train_loss))
            
            _dir = './log/{}/{}'.format(self.dataset_name, self.config["model_type"])
            Analysis.save_same_row_list(_dir, 'train_log', loss=Epoch_loss)
            if resume_path == False:
                self.save_model(epoch, 0)
            else:
                self.save_model(epoch, int(resume_path.split('/')[-1].split('_')[1].split('.')[0]))
            
            if is_eval == True:
                scores, eval_loss = self.eval(epoch, num_epochs, eval_mode=eval_mode, gpu=gpu, save_pred_dir=_dir)
                Epoch_score_eval.append(json.dumps(scores))
                Epoch_loss_eval.append(eval_loss)
                Analysis.save_same_row_list(_dir, 'eval_log', loss=Epoch_loss_eval, scores=Epoch_score_eval)

    def save_model(self, epoch, save_offset=0):
        _dir = './model/{}/{}'.format(self.dataset_name, self.config["model_type"])
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        torch.save(self.model, '{}/epoch_{}.pth'.format(_dir, epoch + 1 + save_offset))

    def eval(self, epoch, num_epochs, resume_path=False, gpu=[0, 1, 2, 3], eval_mode='dev', save_pred_dir=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and self.model_cuda == False:
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
            self.model_cuda = True
            self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(model_dict)
            self.model.to(device)

        with torch.no_grad():
            eval_loss = []
            SRC = []
            TARGET = []
            PREDICTION = []
            eval_count = 0
            self.model.eval()
            eval_iter = tqdm(self.eval_loader) if eval_mode == 'dev' else tqdm(self.test_loader)
            for it in eval_iter:
                self.model.eval()
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                outputs = self.model(**it)
                loss = outputs.loss
                pred = outputs.logits
                loss = loss.mean()

                eval_loss.append(loss.sum().data.item())
                
                cur_SRC = it['input_ids'].tolist()
                cur_TARGET = it['input_ids'].tolist()
                cur_PREDICTION = pred.max(-1)[1].tolist()

                for idx, _ in enumerate(cur_SRC):
                    cur_TARGET[idx] = cur_TARGET[idx][it['src_length'][idx]:]
                    cur_PREDICTION[idx] = cur_PREDICTION[idx][it['src_length'][idx]:]
                
                SRC += cur_SRC
                TARGET += cur_TARGET
                PREDICTION += cur_PREDICTION
                
                eval_count += 1
                
                eval_iter.set_description('Eval: {}/{}'.format(epoch + 1, num_epochs))
                eval_iter.set_postfix(eval_loss=np.mean(eval_loss))
            
            print('Compuing Score ...')
            scores = self.analysis.Evaluation(SRC, TARGET, PREDICTION)
            print('\n', scores)

            if save_pred_dir != False:
                self.analysis.save_xy(TARGET, PREDICTION, save_pred_dir)
            
            return scores, np.mean(eval_loss)
    
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
    
    def eval_without_hint(self, epoch, num_epochs, resume_path=False, gpu=[0, 1, 2, 3], eval_mode='dev', save_pred_dir=False, top_k=8, top_p=0.5):
        self.top_k = top_k
        self.top_p = top_p
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and self.model_cuda == False:
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
            self.model_cuda = True
            self.model.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(model_dict)
            self.model.to(device)
        
        with torch.no_grad():
            SRC = []
            TARGET = []
            PREDICTION = []
            eval_count = 0
            self.model.eval()
            eval_iter = tqdm(self.eval_loader) if eval_mode == 'dev' else tqdm(self.test_loader)
            for it in eval_iter:
                self.model.eval()
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                pred_input = torch.tensor([]).int()
                pred_input = self.cuda(pred_input)
                while len(pred_input) < 100:
                    it['input_ids'] = torch.cat((it['src_input_ids'], pred_input.unsqueeze(0)), dim=-1)
                    it['attention_mask'] = it['input_ids'].gt(50256)
                    it['labels'] = it['input_ids']
                    r = self.model(**it)
                    pred = r.logits
                    next_token_logits = pred[:, -1]
                    filtered_logits = self.top_k_top_p_filtering(
                        next_token_logits[0], top_k=self.top_k, top_p=self.top_p
                    )
                    next_token = torch.multinomial(
                        torch.softmax(filtered_logits, dim=-1), num_samples=1
                    )
                    pred_input = torch.cat([pred_input, next_token], dim=0)
                    eval_iter.set_postfix(length=pred_input.shape[0])
                
                cur_SRC = it['input_ids'].tolist()
                cur_TARGET = it['target_input_ids'].tolist()
                cur_PREDICTION = pred_input.unsqueeze(0).tolist()
                
                SRC += cur_SRC
                TARGET += cur_TARGET
                PREDICTION += cur_PREDICTION
                
                eval_count += 1
                
                eval_iter.set_description('Eval: {}/{}'.format(epoch + 1, num_epochs))
            
            print('Compuing Score ...')
            scores = self.analysis.Evaluation(SRC, TARGET, PREDICTION)
            print('\n', scores)

            if save_pred_dir != False:
                self.analysis.save_xy(TARGET, PREDICTION, save_pred_dir)
            
            return scores
        
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