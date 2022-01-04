import json
import torch
import spacy
import pickle
from tqdm import tqdm
from CC.dataloaders import *
from CC.models.gpt2 import GPT2
from transformers import GPT2Tokenizer

def save_data_entities(entity_file_name, tripple_file_name, tripple_template_file_name, save_file_name):
    e2i_list = []
    i2e_list = []
    with open(entity_file_name) as f:
        entity_list = f.read().split('\n')
    if entity_list[-1] == '':
        entity_list = entity_list[:-1]
        entity_list = [item.strip().split(' <SEP> ') for item in entity_list]
    with open(tripple_file_name) as f:
        tripple_list = f.read().split('\n')
    if tripple_list[-1] == '':
        tripple_list = tripple_list[:-1]
        tripple_list = [item.strip().split('< TSP >') for item in tripple_list]
    with open(tripple_template_file_name) as f:
        template_list = f.read().split('\n')
    if template_list[-1] == '':
        template_list = template_list[:-1]
        template = {}
        for item in template_list:
            item = item.split('\t')
            template[item[0]] = item[1]
    count = len(entity_list)
    for idx in tqdm(range(count)):
        e2i = {}
        i2e = {}
        c = 0
        entities = entity_list[idx]
        tripples = tripple_list[idx]
        for e in entities:
            e2i[e] = c
            i2e[c] = {
                'name': e,
                'context': '',
                'links': [],
                'status': 'init'
            }
            c += 1
        for tripple in tripples:
            a, r, b = tripple.split(' | ')
            a = a.strip()
            r = r.strip()
            b = b.strip()
            if b not in e2i:
                e2i[b] = c
                i2e[c] = {
                    'name': b,
                    'context': '',
                    'links': [],
                    'status': 'init'
                }
                cur_template = template[r]
                context = cur_template.replace('@A', a)
                context = context.replace('@B', b)
                i2e[c]['context'] = context
                c += 1
            if a in e2i:
                if b not in i2e[e2i[a]]['links']:
                    i2e[e2i[a]]['links'].append(b)
        e2i_list.append(e2i)
        i2e_list.append(i2e)
    with open(save_file_name, mode='w+') as f:
        f.write("")
    with open(save_file_name, mode='a+') as f:
        for idx, _ in tqdm(enumerate(e2i_list)):
            f.write('{}\t{}\n'.format(json.dumps(e2i_list[idx]), json.dumps(i2e_list[idx])))

def save_entities_hidden_state(entities_file_name, save_dir, padding_length=512):
    tokenizer = GPT2Tokenizer.from_pretrained('model/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token
    model = GPT2(tokenizer=tokenizer, pretrained_dir='model/gpt2')
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    e_list = ENTDESC_CogNLG_DataloaderGPT2.load_entities_from_file(entities_file_name)
    count = 0
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for item in tqdm(e_list):
        x = {}
        e2i, i2e = item
        for i in i2e:
            content = i2e[i]['name'] if i2e[i]['context'] == '' else i2e[i]['context']
            T = tokenizer(content, add_special_tokens=True, max_length=padding_length, truncation=True)
            input_ids = torch.tensor(T['input_ids']).cuda()
            attn_mask = torch.tensor(T['attention_mask']).cuda()
            with torch.no_grad():
                hidden_states = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids, output_hidden_states=True).hidden_states[-1]
            hidden_states = hidden_states[-1]
            x[i] = hidden_states.cpu()
        with open(os.path.join(save_dir, '{}.hs'.format(count)), mode='wb') as f:
            pickle.dump(x, f, 2)
        count += 1

g_nlp = False

def save_tgt_ners(file_name, save_file_name):
    from torch.multiprocessing import Pool
    global g_nlp
    g_nlp = spacy.load("en_core_web_sm")
    with open(file_name) as f:
        ori_list = f.read().split('\n')
    if ori_list[-1] == '':
        ori_list = ori_list[:-1]
    result = []
    with Pool(30) as pool:
        for r in tqdm(pool.imap(save_tgt_ners_fn, ori_list)):
            result.append(r)
    with open(save_file_name, mode='w+') as f:
        f.write("")
    with open(save_file_name, mode='a+') as f:
        for idx, _ in tqdm(enumerate(result)):
            f.write('{}\n'.format(json.dumps(result[idx])))

def save_tgt_ners_fn(line):
    r = {}
    r['ner'] = []
    doc = g_nlp(line)
    for ent in doc.ents:
        r['ner'].append({
            'text': ent.text, 
            'start_char': ent.start_char, 
            'end_char': ent.end_char,
            'label': ent.label_
        })
    r['tgt'] = line
    return r

def save_triple_type_template(file_names, save_file_name):
    final_list = []
    for file_name in file_names:
        with open(file_name) as f:
            ori_list = f.read().split('\n')
        if ori_list[-1] == '':
            ori_list = ori_list[:-1]
        final_list += ori_list
    result = []
    for item in final_list:
        item = item.strip().split('< TSP >')
        for triple in item:
            triple = triple.split(' | ')
            if triple[1] not in result:
                result.append(triple[1])
    with open(save_file_name, mode='w+') as f:
        f.write("")
    with open(save_file_name, mode='a+') as f:
        for idx, _ in tqdm(enumerate(result)):
            f.write('{}\tthe {} of @A is @B\n'.format(result[idx], result[idx]))