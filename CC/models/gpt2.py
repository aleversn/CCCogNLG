import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2(nn.Module):

    def __init__(self, tokenizer, pretrained_dir):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_dir)
        self.tokenizer = tokenizer
    
    def forward(self, **args):
        if 'output_hidden_states' not in args:
            args['output_hidden_states'] = False
        if 'output_attentions' not in args:
            args['output_attentions'] = False
        if 'labels' not in args:
            args['labels'] = args['input_ids']
        r = self.model(input_ids=args['input_ids'], attention_mask=args['attention_mask'], labels=args['labels'], output_hidden_states=args['output_hidden_states'], output_attentions=args['output_attentions'])
        
        return r