import os
import re
import json
import datetime
import numpy as np
from CC.SQLManager import SQLManager
from CC.ICCStandard import IAnalysis
from CC.evals.bleu.bleu import Bleu
from CC.evals.rouge.rouge import Rouge
from CC.evals.nltk_bleu.nltk_bleu import NLTK_Bleu
from urllib.parse import quote, unquote

class Analysis(IAnalysis):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bleu_scorer = Bleu()
        self.rouge_scorer = Rouge()
        self.nltk_bleu_scorer = NLTK_Bleu()
    
    @staticmethod
    def WriteSDC(name, info):
        with open("./log/{}.txt".format(name), mode="a+", encoding="utf-8") as f:
            f.write(info)
    
    '''
    用于模型生成结果分数计算
    '''
    def Evaluation(self, SRC, TARGET, PREDICTION):
        if len(TARGET) == 0:
            return {'status': 'length is zero.'}
        if len(TARGET) != len(PREDICTION):
            raise Exception('mismatch length of TARGET and PREDICTION.')
        res = {}
        gts = {}
        for idx, src in enumerate(SRC):
            src = self.tokenizer.decode(src, skip_special_tokens=True)
            res[src] = [self.tokenizer.decode(PREDICTION[idx], skip_special_tokens=True)]
            if src in gts:
                gts[src].append(self.tokenizer.decode(TARGET[idx], skip_special_tokens=True))
            else:
                gts[src] = [self.tokenizer.decode(TARGET[idx], skip_special_tokens=True)]
        
        bleu_scores, _ = self.bleu_scorer.compute_score(gts, res)
        nltk_bleu_score = self.nltk_bleu_scorer.compute_score(gts, res)
        rouge_score, _ = self.rouge_scorer.compute_score(gts, res)
        
        bleu_result = {
            'Bleu_1': bleu_scores[0],
            'Bleu_2': bleu_scores[1],
            'Bleu_3': bleu_scores[2],
            'Bleu_4': bleu_scores[3],
            'NLTK_BLEU': nltk_bleu_score,
            'ROUGE_L': rouge_score
        }

        return bleu_result
    
    @staticmethod
    def compute_scores_from_txt(file_name, scorer):
        with open(file_name, encoding='utf-8', mode='r') as f:
            ori_list = f.read().split('\n')
        ori_list = ori_list[:-1]
        tgt = {}
        src = {}
        for idx, item in enumerate(ori_list):
            s, t = item.split('\t')
            tgt[idx] = [t.strip()]
            src[idx] = [s.strip()[:len(t)]]
        
        scorer = scorer
        scores, _ = scorer.compute_score(tgt, src)
        return scores
    
    @staticmethod
    def compute_bleu_from_txt(file_name):
        return Analysis.compute_scores_from_txt(file_name, Bleu())
    
    @staticmethod
    def compute_rouge_from_txt(file_name):
        return Analysis.compute_scores_from_txt(file_name, Rouge())
    
    @staticmethod
    def compute_nltk_bleu_from_txt(file_name):
        return Analysis.compute_scores_from_txt(file_name, NLTK_Bleu())
    
    @staticmethod
    def compute_scores_of_each_row_from_txt(scorer, file_name, save_file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
            ori_list = f.read().split('\n')
        ori_list = ori_list[:-1]
        result = []
        for idx, item in enumerate(ori_list):
            tgt = {}
            src = {}
            s, t = item.split('\t')
            tgt[idx] = [t.strip()]
            src[idx] = [s.strip()[:len(t)]]
        
            scorer = scorer
            scores, _ = scorer.compute_score(tgt, src)
            result.append(scores)
        
        with open(save_file_name, encoding='utf-8', mode='w+') as f:
            f.write('')
        with open(save_file_name, encoding='utf-8', mode='a+') as f:
            for r in result:
                f.write('{}\n'.format(json.dumps(r)))
        return result
    
    @staticmethod
    def compute_bleu_of_each_row_from_txt(file_name, save_file_name):
        return Analysis.compute_scores_of_each_row_from_txt(Bleu(), file_name, save_file_name)
    
    @staticmethod
    def compute_rouge_of_each_row_from_txt(file_name, save_file_name):
        return Analysis.compute_scores_of_each_row_from_txt(Rouge(), file_name, save_file_name)
    
    @staticmethod
    def heatmap(data):
        return ValueError('')
    
    '''
    保存预测和参考结果
    '''
    def save_xy_with_best_nodes(self, TARGET, PREDICTION, BESTNODES, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        result_pred = ''
        result_gold = ''
        result_pred_idx = ''
        result_best_node_idx = ''
        for idx, _ in enumerate(TARGET):
            p = self.tokenizer.decode(PREDICTION[idx], skip_special_tokens=True)
            p = p.replace('\n', '')
            p = re.sub(' +', ' ', p)
            t = self.tokenizer.decode(TARGET[idx], skip_special_tokens=True)
            result += '{}\t{}\n'.format(p, t)
            result_pred += '{}\n'.format(p)
            result_gold += '{}\n'.format(t)
            result_pred_idx += '{}\n'.format(str(PREDICTION[idx]))
            result_best_node_idx += '{}\n'.format(str(BESTNODES[idx]))
        with open('{}/predict_gold.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result)
        with open('{}/predict.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result_pred)
        with open('{}/gold.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result_gold)
        with open('{}/pred_idx.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result_pred_idx)
        with open('{}/best_node_idx.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result_best_node_idx)
    
    def save_xy(self, TARGET, PREDICTION, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        result_pred = ''
        result_gold = ''
        result_pred_idx = ''
        for idx, _ in enumerate(TARGET):
            p = self.tokenizer.decode(PREDICTION[idx], skip_special_tokens=True)
            p = p.replace('\n', '')
            p = re.sub(' +', ' ', p)
            t = self.tokenizer.decode(TARGET[idx], skip_special_tokens=True)
            result += '{}\t{}\n'.format(p, t)
            result_pred += '{}\n'.format(p)
            result_gold += '{}\n'.format(t)
            result_pred_idx += '{}\n'.format(str(PREDICTION[idx]))
        with open('{}/predict_gold.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result)
        with open('{}/predict.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result_pred)
        with open('{}/gold.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result_gold)
        with open('{}/pred_idx.csv'.format(dir), encoding='utf-8', mode='w+') as f:
            f.write(result_pred_idx)
    
    @staticmethod
    def save_same_row_list(dir, file_name, **args):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        dicts = []
        for key in args.keys():
            dicts.append(key)
            result = key if result == '' else result + '\t{}'.format(key)
        length = len(args[dicts[0]])
        result += '\n'
        for i in range(length):
            t = True
            for key in args.keys():
                result += str(args[key][i]) if t else '\t{}'.format(args[key][i])
                t = False
            result += '\n'
        with open('{}/{}.csv'.format(dir, file_name), encoding='utf-8', mode='w+') as f:
            f.write(result)
    
    def database_init(config_file_name):
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
    
    @staticmethod
    def get_entity_info_from_sql(name):
        name = quote(name)
        sl = Analysis.database_init('./sql_auth')
        info = sl.Get("select * from t_entities where name like '%{}%'".format(name))
        info = [unquote(str(item)) for item in info[0]]
        return info