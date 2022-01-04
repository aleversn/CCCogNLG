import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from CC.ICCStandard import IDataLoader
from CC.dataloaders import *

class AutoDataloader(IDataLoader):

    def __init__(self, tokenizer, data_name, model_type="prophetnet", padding_length=128):
        self.model_type = model_type
        self.model_series_1 = ['prophetnet']
        self.model_series_2 = ['gpt2']
        if data_name == 'qg':
            if model_type in self.model_series_1:
                self.training_set = QGDataloader(tokenizer, src_file_name='datasets/qg_data/prophetnet_tokenized/train.src', tgt_file_name='datasets/qg_data/prophetnet_tokenized/train.tgt', padding_length=padding_length)
                self.dev_set = QGDataloader(tokenizer, src_file_name='datasets/qg_data/prophetnet_tokenized/dev.src', tgt_file_name='datasets/qg_data/prophetnet_tokenized/dev.tgt', padding_length=padding_length, shuffle=False)
                self.test_set = QGDataloader(tokenizer, src_file_name='datasets/qg_data/prophetnet_tokenized/test.src', tgt_file_name='datasets/qg_data/prophetnet_tokenized/test.tgt', padding_length=padding_length, shuffle=False)
            elif model_type in self.model_series_2:
                self.training_set = QGDataloaderGPT2(tokenizer, src_file_name='datasets/qg_data/prophetnet_tokenized/train.src', tgt_file_name='datasets/qg_data/prophetnet_tokenized/train.tgt', padding_length=padding_length, mode='train')
                self.dev_set = QGDataloaderGPT2(tokenizer, src_file_name='datasets/qg_data/prophetnet_tokenized/dev.src', tgt_file_name='datasets/qg_data/prophetnet_tokenized/dev.tgt', padding_length=padding_length, shuffle=False, mode='eval')
                self.test_set = QGDataloaderGPT2(tokenizer, src_file_name='datasets/qg_data/prophetnet_tokenized/test.src', tgt_file_name='datasets/qg_data/prophetnet_tokenized/test.tgt', padding_length=padding_length, shuffle=False, mode='test')
        elif data_name == 'cnn':
            if model_type in self.model_series_1:
                self.training_set = CNN_1024Dataloader(tokenizer, file_dir='datasets/CNN/gpt2_1024_data', padding_length=padding_length)
                self.dev_set = CNN_1024Dataloader(tokenizer, file_dir='datasets/CNN/gpt2_1024_data', padding_length=padding_length, shuffle=False)
                self.test_set = CNN_1024Dataloader(tokenizer, file_dir='datasets/CNN/gpt2_1024_data', padding_length=padding_length, shuffle=False)
            elif model_type in self.model_series_2:
                self.training_set = CNN_1024Dataloader(tokenizer, file_dir='datasets/CNN/gpt2_1024_data', padding_length=padding_length, mode='train')
                self.dev_set = CNN_1024Dataloader(tokenizer, file_dir='datasets/CNN/gpt2_1024_data', padding_length=padding_length, shuffle=False, mode='eval')
                self.test_set = CNN_1024Dataloader(tokenizer, file_dir='datasets/CNN/gpt2_1024_data', padding_length=padding_length, shuffle=False, mode='test')
        elif data_name == 'entdesc':
            if model_type in self.model_series_2:
                self.training_set = ENTDESC_DataloaderGPT2(tokenizer, src_file_name='./datasets/ENT-DESC/train-entity.txt', tgt_file_name='datasets/ENT-DESC/train_surface.pp.txt', padding_length=padding_length)
                self.dev_set = ENTDESC_DataloaderGPT2(tokenizer, src_file_name='./datasets/ENT-DESC/dev-entity.txt', tgt_file_name='datasets/ENT-DESC/dev_surface.pp.txt', padding_length=padding_length, shuffle=False)
                self.test_set = ENTDESC_DataloaderGPT2(tokenizer, src_file_name='./datasets/ENT-DESC/test-entity.txt', tgt_file_name='datasets/ENT-DESC/test_surface.pp.txt', padding_length=padding_length, shuffle=False)
        elif data_name == 'entdesc_triples':
            if model_type in self.model_series_2:
                self.training_set = ENTDESC_DataloaderGPT2_Triple(tokenizer, src_file_name='./datasets/ENT-DESC/train-triples.txt', tgt_file_name='datasets/ENT-DESC/train_surface.pp.txt', padding_length=padding_length)
                self.dev_set = ENTDESC_DataloaderGPT2_Triple(tokenizer, src_file_name='./datasets/ENT-DESC/dev-triples.txt', tgt_file_name='datasets/ENT-DESC/dev_surface.pp.txt', padding_length=padding_length, shuffle=False)
                self.test_set = ENTDESC_DataloaderGPT2_Triple(tokenizer, src_file_name='./datasets/ENT-DESC/test-triples.txt', tgt_file_name='datasets/ENT-DESC/test_surface.pp.txt', padding_length=padding_length, shuffle=False)
        elif data_name == 'entdesc_cognlg':
            if model_type in self.model_series_2:
                self.training_set = ENTDESC_CogNLG_DataloaderGPT2(tokenizer, src_file_name='./datasets/ENT-DESC/train-entity.txt', tgt_file_name='datasets/ENT-DESC/train_surface.pp.txt', padding_length=padding_length)
                self.dev_set = ENTDESC_CogNLG_DataloaderGPT2(tokenizer, src_file_name='./datasets/ENT-DESC/dev-entity.txt', tgt_file_name='datasets/ENT-DESC/dev_surface.pp.txt', padding_length=padding_length, shuffle=False)
                self.test_set = ENTDESC_CogNLG_DataloaderGPT2(tokenizer, src_file_name='./datasets/ENT-DESC/test-entity.txt', tgt_file_name='datasets/ENT-DESC/test_surface.pp.txt', padding_length=padding_length, shuffle=False)
    
    def __call__(self, batch_size=16, batch_size_eval=64):
        dataiter = DataLoader(self.training_set, batch_size=batch_size)
        dataiter_eval = DataLoader(self.dev_set, batch_size=batch_size_eval)
        dataiter_test = DataLoader(self.test_set, batch_size=batch_size_eval)
        return dataiter, dataiter_eval, dataiter_test