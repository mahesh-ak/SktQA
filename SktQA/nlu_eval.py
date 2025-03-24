import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import MAX_K, DEFAULT_MODELS, LOW_END_MODELS
import string
import evaluate as ev
import json
from collections import Counter
from tqdm import tqdm
from math import ceil

MODELS = LOW_END_MODELS + DEFAULT_MODELS
punct_table = str.maketrans(dict.fromkeys(string.punctuation))
bleu = ev.load('sacrebleu')
seqeval = ev.load('seqeval')
NUM_CHUNKS = 10

def compare(ans_list,y):
    return str(y).replace('उत्तरम्','').translate(punct_table).strip() in [x.replace('।','').translate(punct_table).strip() for x in ans_list.split(';')]

def format_ner(dict_json, sent):
    words = sent.split()
    try:
        dict_json = json.loads(dict_json.replace("'",'"'))
    except:
        dict_json = {}
    inv_dict = {}
    if not type(dict_json) == dict:
        dict_json = {}
    for k, v in dict_json.items():
        for vi in v:
            inv_dict[vi] = k
    return [inv_dict[w] if w in inv_dict else 'O' for w in words]

SKT_ENT = ['ASURA', 'RAKSHASA', 'HUMAN', 'KULA', 'DEVA', 'PALACE', 'NAGA', 'GANDHARVA', 'TREE', 'FLOWER', 'MOUNTAIN', 'KINGDOM', 'VANARA', 'AXE', 'ORNAMENT', 'MUHURTA', 'SEA', 'HOUSE', 'GARDEN', 'FOREST', 'ASTRA', 'VINE', 'RIVERBANK', 'GRAHA', 'CITY', 'GRIDHRA', 'ARROW', 'ROAD', 'FESTIVAL', 'SWARGA', 'FRUIT', 'RATHA']
LAT_ENT = ['PERS', 'LOC', 'GRP']
GRA_ENT = ['LOC', 'GOD', 'ORG', 'NORP', 'WORK', 'EVENT', 'PERSON', 'LANGUAGE']

def eval_file_ner(in_file):
    df_ = pd.read_csv(in_file, sep='\t')
    N = len(df_)
    step = ceil(N/NUM_CHUNKS)
    if 'gold' not in df_.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    
    methods = MODELS 
    scores = {m: [] for m in methods}
    
    if 'skt' in in_file:
        ent = SKT_ENT
    elif 'lat' in in_file:
        ent = LAT_ENT
    elif 'gra' in in_file:
        ent = GRA_ENT
    print('Computing NER scores from',in_file)
    for i in tqdm(range(0,N,step)):
        df = df_[i:min(N,i+step)]
        references = [ref.split() for ref in df['gold'].tolist()]
        all_refs = []
        for ref in references:
            all_refs = all_refs + [c.replace('B-','').replace('I-','') for c in ref]
        for m in methods:
            df[m] = df.apply(lambda x: format_ner(x[m],x['sentence']), axis = 1)
            predictions = df[m].tolist()
            scores_ = seqeval.compute(predictions=predictions, references=references)
            F1 = [v['f1'] for k,v in scores_.items() if k in ent]
            scores[m].append(np.mean(F1))

    return scores

def eval_file_mt(in_file):
    df_ = pd.read_csv(in_file, sep='\t')
    if 'mt' in in_file:
        df_2 = pd.read_csv(in_file.replace('in','out'),sep='\t')
        df_ = pd.concat([df_,df_2])
    if 'gold' not in df_.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    
    methods = MODELS 
    scores = {m:[] for m in methods}
    N = len(df_)
    step = ceil(N/NUM_CHUNKS)

    print('Computing MT scores from',in_file)
    for i in tqdm(range(0,N,step)):
        df = df_[i:min(N,i+step)]
        references = [[ref] for ref in df['gold'].tolist()]
        for m in methods:
            predictions = df[m].tolist()
            scores[m].append(bleu.compute(predictions=predictions, references=references)['score']/100)
    return scores

def eval_file_qa(df_):
    if 'ANSWER' not in df_.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    
    methods = MODELS 
    em_scores = {m: [] for m in methods}
    N = len(df_)
    step = ceil(N/NUM_CHUNKS)

    print('Computing QA scores')
    for i in tqdm(range(0,N,step)):
        df = df_[i:min(N,i+step)]
        for m in methods:
            em = df.apply(lambda x: compare(x['ANSWER'], x[m]), axis=1)
            em_scores[m].append(em.sum()/len(em))

    return em_scores

def eval_file_rel(df_, rel_df_, reverse = False):
    rel_df_ = rel_df_[['ID','rel_0','rel_1','rel_2', 'rel_3']]
    rel_df_['rel_sum'] = rel_df_.apply(lambda x: sum([x[f'rel_{k}'] for k in range(4)]),axis=1)

    if reverse:
        rel_df_ = rel_df_[rel_df_['rel_sum']==0][['ID']]
    else:
        rel_df_ = rel_df_[rel_df_['rel_sum']>0][['ID']]

    df_ = df_.merge(rel_df_, how='inner')
    print('Length of relavant rows', len(df_))
    if 'ANSWER' not in df_.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    methods = MODELS
    em_scores = {m: [] for m in MODELS}
    N = len(df_)
    step = ceil(N/NUM_CHUNKS)

    print('Computing QA rel scores')
    for i in tqdm(range(0,N,step)):
        df = df_[i:min(N,i+step)]
        for m in methods:
            em = df.apply(lambda x: compare(x['ANSWER'], x[m]), axis=1)
            em_scores[m].append(em.sum()/len(em))

    return em_scores



def eval_default():
    results = {}

    ## NER evaluation
    lang = {'san': 'skt_ner', 'lat': 'lat_ner', 'grc': 'gra_ner'}
    f_pth = "results/ner/{lang}_{n}.tsv"
    scores = {}
    for l in lang:
        l_f_pth = f_pth.format(lang=lang[l], n=0)
        scores[l] = eval_file_ner(l_f_pth)

    results['(a) Named Entity Recognition'] = scores

    ## MT evaluation
    lang = {'san': 'mt_in', 'lat': 'lat_eng', 'grc':'grc_eng'}
    f_pth = "results/mt/{lang}_{n}.tsv"
    scores = {}
    for l in lang:
        l_f_pth = f_pth.format(lang=lang[l],n=0)
        scores[l] = eval_file_mt(l_f_pth)
    
    results['(b) Machine Translation to English'] = scores

    ## QA evaluation
    files = {'w/o context': [f"results/zero_shot/{pre}_0.tsv" for pre in ['sanskrit','ayurveda']],
            '+ context (RAG-BM25)': [f"results/rag/{pr}bm25_4.tsv" for pr in ['','ayurveda_']]}
    scores = {}

    rel_files = [f"data/{pr}bm25_4_rel.tsv" for pr in ['','ayurveda_']]

    rel_dfs = [pd.read_csv(rf,sep='\t') for rf in rel_files]
    for i in range(2):
        rel_dfs[i]['ID'] = rel_dfs[i].apply(lambda x: f"{x['ID']}_{i}",axis=1)
    rel_df = pd.concat(rel_dfs)


    rel_scores = {}
    rel_scores_neg = {}
    for f in files:
        df_ = pd.read_csv(files[f][0], sep='\t')
        df_['ID'] = df_.apply(lambda x: f"{x['ID']}_0", axis=1)
        df_2 = pd.read_csv(files[f][1], sep='\t')
        df_2['ID'] = df_2.apply(lambda x: f"{x['ID']}_1", axis=1)
        df_ = pd.concat([df_, df_2])
        scores[f] = eval_file_qa(df_)
        rel_scores[f] = eval_file_rel(df_, rel_df)
        rel_scores_neg[f] = eval_file_rel(df_, rel_df, reverse=True)
    
    results['(c) Question Answering (san) - Overall'] = scores
    results['(d) Question Answering (san) - Answer in Context'] = rel_scores
    results['(e) Question Answering (san) - Answer not in Context'] = rel_scores_neg
    with open("results.json",'w') as fp:
        json.dump(results, fp, indent='\t')


if __name__=='__main__':
    eval_default()

