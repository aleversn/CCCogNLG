import os
import sys
import json
import torch
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import ProphetNetTokenizer
from CC.ICCStandard import IDataLoader
from urllib.parse import quote, unquote
from CC.SQLManager import SQLManager


class QGDataloader(Dataset):

    def __init__(self, tokenizer: ProphetNetTokenizer, src_file_name: str, tgt_file_name: str, padding_length=128, shuffle=True):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(src_file_name, tgt_file_name)
        if shuffle:
            random.shuffle(self.ori_list)
    
    def load_train(self, src_file_name: str, tgt_file_name: str):
        with open(src_file_name, encoding='utf-8') as f:
            src_list = f.read().split('\n')
        if src_list[len(src_list) - 1] == '':
            src_list = src_list[:len(src_list) - 1]
        with open(tgt_file_name, encoding='utf-8') as f:
            tgt_list = f.read().split('\n')
        if tgt_list[len(tgt_list) - 1] == '':
            tgt_list = tgt_list[:len(tgt_list) - 1]
        train_list = [(item, tgt_list[idx]) for idx, item in enumerate(src_list)]
        return train_list
    
    def __getitem__(self, idx):
        line = self.ori_list[idx]
        src, tgt = line[0], line[1]
        T_src = self.tokenizer(src, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        src_input_ids = torch.tensor(T_src['input_ids'])
        src_attn_mask = torch.tensor(T_src['attention_mask'])
        T_tgt = self.tokenizer(tgt, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        tgt_input_ids = torch.tensor(T_tgt['input_ids'])
        tgt_attn_mask = torch.tensor(T_tgt['attention_mask'])
        return {
            'input_ids': src_input_ids,
            'attention_mask': src_attn_mask,
            'decoder_input_ids': tgt_input_ids,
            'decoder_attention_mask': tgt_attn_mask
        }
    
    def __len__(self):
        return len(self.ori_list)

class QGDataloaderGPT2(QGDataloader):

    def __init__(self, tokenizer, src_file_name: str, tgt_file_name: str, padding_length=128, shuffle=True, mode='train'):
        super(QGDataloaderGPT2, self).__init__(tokenizer, src_file_name, tgt_file_name, padding_length, shuffle)
        self.tokenizer.pad_token = tokenizer.eos_token
        self.mode = mode
    
    def __getitem__(self, idx):
        line = self.ori_list[idx]
        src, tgt = line[0], line[1]
        T = self.tokenizer(src + self.tokenizer.sep_token + tgt, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])

        T_src = self.tokenizer(src, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        src_input_ids = torch.tensor(T_src['input_ids'])
        src_attn_mask = torch.tensor(T_src['attention_mask'])

        T_tgt = self.tokenizer(tgt, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        tgt_input_ids = torch.tensor(T_tgt['input_ids'])
        tgt_attn_mask = torch.tensor(T_tgt['attention_mask'])
        
        src_length = len(self.tokenizer(src)['input_ids'])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attn_mask,
            'target_input_ids': tgt_input_ids,
            'target_attention_mask': tgt_attn_mask,
            'src_length': src_length
        }

class CNN_1024Dataloader(Dataset):
    def __init__(self, tokenizer, file_dir, padding_length=128, shuffle=True, mode='train'):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.id_list = os.listdir(file_dir)
        self.file_dir = file_dir
        self.mode = mode
    
    def __getitem__(self, idx):
        if self.mode != 'train':
            idx = self.id_list[-idx]
        else:
            idx = self.id_list[idx]
            
        file_name = os.path.join(self.file_dir,str(idx))
        with open(file_name,'r') as f:
              data = json.load(f)
        src, tgt = self.tokenizer.decode(data['article']), self.tokenizer.decode(data['abstract'])
        T = self.tokenizer(src + self.tokenizer.sep_token + tgt, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        
        T_src = self.tokenizer(src, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        src_input_ids = torch.tensor(T_src['input_ids'])
        src_attn_mask = torch.tensor(T_src['attention_mask'])
        
        T_tgt = self.tokenizer(tgt, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        tgt_input_ids = torch.tensor(T_tgt['input_ids'])
        tgt_attn_mask = torch.tensor(T_tgt['attention_mask'])

        src_length = len(self.tokenizer(src)['input_ids'])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attn_mask,
            'target_input_ids': tgt_input_ids,
            'target_attention_mask': tgt_attn_mask,
            'src_length': src_length
        }
    
    def __len__(self):
        length = len(self.id_list)
        if self.mode != 'train':
            return int(0.2 * length)
        else:
            return int(0.8 * length)

class ENTDESC_DataloaderGPT2(Dataset):

    def __init__(self, tokenizer, src_file_name: str, tgt_file_name: str, padding_length=128, shuffle=True, mode='train'):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(src_file_name, tgt_file_name)
        self.tokenizer.pad_token = tokenizer.eos_token
        self.pad_token = tokenizer(tokenizer.pad_token)['input_ids'][0]
        self.mode = mode
    
    def load_train(self, src_file_name, tgt_file_name):
        with open(src_file_name, encoding='utf-8') as f:
            src_list = f.read().split('\n')
        if src_list[len(src_list) - 1] == '':
            src_list = src_list[:len(src_list) - 1]
        with open(tgt_file_name, encoding='utf-8') as f:
            tgt_list = f.read().split('\n')
        if tgt_list[len(tgt_list) - 1] == '':
            tgt_list = tgt_list[:len(tgt_list) - 1]
        ori_list = [(src_list[idx], tgt_list[idx]) for idx, _ in enumerate(tgt_list)]
        return ori_list
    
    def __getitem__(self, idx):
        line = self.ori_list[idx]
        src, tgt = line[0], line[1]
        src = src.replace('<SEP>', self.tokenizer.eos_token)

        T_src = self.tokenizer(src + self.tokenizer.eos_token, add_special_tokens=True, max_length=int(self.padding_length / 2), truncation=True)
        src_input_ids = torch.tensor(T_src['input_ids'])
        src_attn_mask = torch.tensor(T_src['attention_mask'])

        T_tgt = self.tokenizer(tgt, add_special_tokens=True, max_length=int(self.padding_length / 2),truncation=True)
        tgt_input_ids = torch.tensor(T_tgt['input_ids'])
        tgt_attn_mask = torch.tensor(T_tgt['attention_mask'])
        
        src_length = len(src_input_ids)
        tgt_length = len(tgt_input_ids)

        input_ids = torch.cat([src_input_ids, tgt_input_ids])
        remain = torch.tensor([self.pad_token for i in range(self.padding_length - len(input_ids))])
        input_ids = torch.cat([input_ids, remain])
        attn_mask = input_ids.gt(self.pad_token)

        labels = input_ids.clone()
        labels[:src_length - 1] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attn_mask,
            'target_input_ids': tgt_input_ids,
            'target_attention_mask': tgt_attn_mask,
            'src_length': src_length,
            'tgt_length': tgt_length,
            'labels': labels
        }
    
    def __len__(self):
        return len(self.ori_list)

class ENTDESC_DataloaderGPT2_Triple(ENTDESC_DataloaderGPT2):
    def __getitem__(self, idx):
        self.mask_id = 50256
        line = self.ori_list[idx]
        src, tgt = line[0], line[1]
        src = src.replace('< TSP >', self.tokenizer.eos_token)

        T_src = self.tokenizer(src, add_special_tokens=True, max_length=self.padding_length, truncation=True)
        src_input_ids = T_src['input_ids']
        remain = [self.mask_id for i in range(self.padding_length - len(src_input_ids))]
        src_input_ids = src_input_ids + remain
        src_input_ids = torch.tensor(src_input_ids)
        src_attn_mask = src_input_ids.gt(self.mask_id)

        T_tgt = self.tokenizer(tgt, add_special_tokens=True, max_length=self.padding_length, truncation=True)
        tgt_input_ids = T_tgt['input_ids']
        remain = [self.mask_id for i in range(self.padding_length - len(tgt_input_ids))]
        tgt_input_ids = tgt_input_ids + remain
        tgt_input_ids = torch.tensor(tgt_input_ids)
        tgt_attn_mask = tgt_input_ids.gt(self.mask_id)
        
        src_length = len(self.tokenizer(src)['input_ids'])
        tgt_length = len(self.tokenizer(tgt)['input_ids'])
        
        input_ids = T_src['input_ids'][:int(self.padding_length / 2)] + T_tgt['input_ids'][:int(self.padding_length / 2)]
        remain = [self.mask_id for i in range(self.padding_length - len(input_ids))]
        input_ids = input_ids + remain
        input_ids = torch.tensor(input_ids)
        attn_mask = input_ids.gt(self.mask_id)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attn_mask,
            'target_input_ids': tgt_input_ids,
            'target_attention_mask': tgt_attn_mask,
            'src_length': src_length,
            'tgt_length': tgt_length
        }

class ENTDESC_CogNLG_DataloaderGPT2(Dataset):

    def __init__(self, tokenizer, src_file_name: str, tgt_file_name: str, padding_length=128, shuffle=True, mode='train'):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(src_file_name, tgt_file_name)
        self.tokenizer.pad_token = tokenizer.eos_token
        self.sl = self.database_init('./sql_auth')
        self.shuffle = shuffle
        self.mode = mode
        if self.shuffle == True:
            self.random_idx = [i for i in range(len(self.ori_list))]
            random.shuffle(self.random_idx)
    
    def database_init(self, config_file_name):
        '''
        {
            "host": "",
            "username": "",
            "password": "",
            "databasename": ""
        }
        '''
        with open(config_file_name, encoding='utf-8', mode='r') as f:
            config = f.read()
        config = json.loads(config)
        sl = SQLManager(config['databasename'], config['username'], config['password'], config['host'])
        return sl
    
    def load_train(self, src_file_name, tgt_file_name):
        with open(src_file_name, encoding='utf-8') as f:
            src_list = f.read().split('\n')
        if src_list[len(src_list) - 1] == '':
            src_list = src_list[:len(src_list) - 1]
        with open(tgt_file_name, encoding='utf-8') as f:
            tgt_list = f.read().split('\n')
        if tgt_list[len(tgt_list) - 1] == '':
            tgt_list = tgt_list[:len(tgt_list) - 1]
        ori_list = [(src_list[idx], tgt_list[idx]) for idx, _ in enumerate(tgt_list)]
        return ori_list
    
    '''
    装载实体以及关联外部实体信息
    '''
    def load_entities(self, src_entities, hops=3):
        e2i = {}
        i2e = {}
        count = 0
        max_nodes = ((3 ** hops - 1) // 2) * len(src_entities)
        for entity in src_entities:
            if entity in e2i:
                continue
            entity_info = self.sl.Get("select * from t_entities where name = '{}';".format(quote(entity)))
            if len(entity_info) == 0:
                entity_info = self.sl.Get("select * from t_entities where name like '%{}%';".format(quote(entity)))
            if len(entity_info) == 0:
                entity_info = [['' for i in range(6)]]
            context = unquote(entity_info[0][2])
            links = [] if unquote(entity_info[0][3]).strip() == '' else unquote(entity_info[0][3]).strip().split(';')[:3]
            if count >= max_nodes:
                links = []
            e2i[entity] = count
            i2e[count] = {
                'name': entity,
                'context': context,
                'links': links,
                'status': 'init'
            }
            src_entities += links
            count += 1
        
        return e2i, i2e
    
    '''
    获取索引下的所有实体
    '''
    def get_entities(self, idx):
        line = self.ori_list[idx]
        src, tgt = line[0], line[1]
        return src.split(' <SEP> ')
    
    '''
    构造邻接矩阵
    '''
    def construct_adjacent(self, e2i, i2e, max_size=512):
        A = torch.eye(max_size)
        for i in i2e:
            e = i2e[i]
            a = int(i)
            links = e['links']
            for e_l in links:
                b = e2i[e_l]
                A[a][b] = 1
        
        # AD-1
        A /= torch.sum(A, dim=0, keepdim=True)
        return A
    
    '''
    从已保存的节点树文件读取节点信息
    '''
    @staticmethod
    def load_entities_from_file(file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
            e_list = f.read().split('\n')
        if e_list[-1] == '':
            e_list = e_list[:-1]
        result = []
        for line in e_list:
            line = line.split('\t')
            result.append((json.loads(line[0]), json.loads(line[1])))
        return result
    
    '''
    从已保存的最佳节点信息读取最佳节点
    '''
    @staticmethod
    def load_best_nodes_from_file(file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
            n_list = f.read().split('\n')
        if n_list[-1] == '':
            n_list = n_list[:-1]
        result = []
        for line in n_list:
            line = json.loads(line)
            line.sort(key=lambda x: x['start'])
            result.append(line)
        return result
    
    def __getitem__(self, idx):
        if self.shuffle == True:
            idx = self.random_idx[idx]
        line = self.ori_list[idx]
        src, tgt = line[0], line[1]
        src = src.replace('<SEP>', self.tokenizer.eos_token)
        T = self.tokenizer(src + self.tokenizer.sep_token + tgt, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])

        T_src = self.tokenizer(src, add_special_tokens=True, max_length=self.padding_length, truncation=True)
        src_input_ids = torch.tensor(T_src['input_ids'])
        src_attn_mask = torch.tensor(T_src['attention_mask'])

        T_tgt = self.tokenizer(tgt, add_special_tokens=True, max_length=self.padding_length, truncation=True)
        tgt_input_ids = torch.tensor(T_tgt['input_ids'])
        tgt_attn_mask = torch.tensor(T_tgt['attention_mask'])
        
        src_length = len(self.tokenizer(src)['input_ids'])
        tgt_length = len(self.tokenizer(tgt)['input_ids'])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attn_mask,
            'target_input_ids': tgt_input_ids,
            'target_attention_mask': tgt_attn_mask,
            'src_length': src_length,
            'tgt_length': tgt_length,
            'idx': idx
        }
    
    def __len__(self):
        return len(self.ori_list)