import json
from CC.ICCStandard import IModel
from CC.models.prophetnet import ProphetNet
from CC.models.gpt2 import GPT2

class AutoModel(IModel):

    def __init__(self, tokenizer, model_pretrained_dir):
        self.tokenizer = tokenizer
        self.config = self.load_model_config(model_pretrained_dir)
        self.model_name = self.config["model_type"]
        self.load_model(model_pretrained_dir)
    
    def load_model_config(self, model_pretrained_dir):
        with open('{}/config.json'.format(model_pretrained_dir), encoding='utf-8') as f:
            config = f.read()
        config = json.loads(config)
        return config
        
    def load_model(self, model_pretrained_dir):
        if self.model_name == 'prophetnet':
            self.model = ProphetNet(tokenizer=self.tokenizer, pretrained_dir=model_pretrained_dir)
        elif self.model_name == 'gpt2':
            self.model = GPT2(tokenizer=self.tokenizer, pretrained_dir=model_pretrained_dir)
    
    def get_model(self):
        return self.model
    
    def __call__(self):
        return self.get_model()
