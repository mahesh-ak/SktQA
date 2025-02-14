import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import MAX_K, DEFAULT_MODELS, LOW_END_MODELS
import string
import evaluate as ev
import json
from collections import Counter
from sklearn.metrics import confusion_matrix

punct_table = str.maketrans(dict.fromkeys(string.punctuation))
bleu = ev.load('sacrebleu')
seqeval = ev.load('seqeval')
def compare(ans_list,y):
    return str(y).replace('उत्तरम्','').translate(punct_table).strip() in [x.replace('।','').translate(punct_table).strip() for x in ans_list.split(';')]

def plot_k(data, pre):
    plt.figure()
    design = {
     'fontsize' : 11,
     'weight' : 'bold'
    }
    map = {'bm25': 'BM25', 'fasttext': 'AvgFT', 'glove': 'AvgGV'}
    for key, values in data.items():
        plt.plot(values.keys(), values.values(), label=map[key], marker='o', linewidth=3)
    
    if pre=='':
        plt.title('Effect of k value in RAG for GPT-4o (Rāmāyaṇa)', **design)
        
        zs_df = pd.read_csv('results/zero_shot/eval_table.tsv', sep='\t')
        txt = dict(zip(zs_df['LLM'], zs_df['sanskrit']))['gpt-4o']

        err = float(txt.split('(')[1].replace(')',''))
        avg = float(txt.split('(')[0].strip())

    elif pre=='ayurveda_':
        plt.title('Effect of k value in RAG for GPT-4o (Bhāvaprakāśanighaṇtu)', **design)
        
        zs_df = pd.read_csv('results/zero_shot/eval_table.tsv', sep='\t')
        txt = dict(zip(zs_df['LLM'], zs_df['ayurveda']))['gpt-4o']

        err = float(txt.split('(')[1].replace(')',''))
        avg = float(txt.split('(')[0].strip())


    plt.axhline(y = avg, color = 'k', linestyle = '-', label='Zero-Shot baseline', linewidth=3) 
    plt.axhline(y = avg+err, color = 'k', linestyle = '--') 
    plt.axhline(y = avg-err, color = 'k', linestyle = '--') 

    # Labels and title
    plt.xlabel('k', **design)
    plt.ylabel('EM score', **design)
    plt.ylim((0.32, 0.55))
    plt.xticks(range(1,MAX_K+1))
    plt.legend(prop={'weight':'bold'})
    plt.grid(True)

    # Show plot
    plt.savefig(f'results/rag/rag_k_plot_{pre}gpt4o.png',bbox_inches='tight',transparent=True, pad_inches=0)

def eval_file_rel(in_file, rel_file, reverse = False):
    df = pd.read_csv(in_file, sep='\t')
    rel_df = pd.read_csv(rel_file,sep='\t')
    if 'rel_0' in rel_df.columns:
        rel_df = rel_df[['ID','rel_0','rel_1','rel_2', 'rel_3']]
        rel_df['rel_sum'] = rel_df.apply(lambda x: sum([x[f'rel_{k}'] for k in range(4)]),axis=1)
    else:
        rel_df['rel_sum'] = rel_df['rel']

    if reverse:
        rel_df = rel_df[rel_df['rel_sum']==0][['ID']]
    else:
        rel_df = rel_df[rel_df['rel_sum']>0][['ID']]

    df = df.merge(rel_df, how='inner')
    print('Length of relavant rows', len(df))
    if 'ANSWER' not in df.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    methods = [col for col in df.columns if (col not in ['QUESTION','ANSWER','ID','CHOICES']) and ('context' not in col) and ('maxd' not in col) and ('paths' not in col)]
    em_scores = {}
    print("Final length:",len(df))
    for m in methods:
        em = df.apply(lambda x: compare(x['ANSWER'], x[m]), axis=1)
        em_scores[m] = round(em.sum()/len(em), 3)
    return em_scores

def eval_file(in_file):
    df = pd.read_csv(in_file, sep='\t')
    if 'ANSWER' not in df.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    
    methods = [col for col in df.columns if (col not in ['QUESTION','ANSWER','ID','CHOICES']) and ('context' not in col) and ('maxd' not in col) and ('paths' not in col)]
    em_scores = {}
    for m in methods:
        em = df.apply(lambda x: compare(x['ANSWER'], x[m]), axis=1)
        em_scores[m] = round(em.sum()/len(em), 3)
    return em_scores

def eval_file_mt(in_file):
    df = pd.read_csv(in_file, sep='\t')
    if 'mt' in in_file:
        df_2 = pd.read_csv(in_file.replace('in','out'),sep='\t')
        df = pd.concat([df,df_2])
    if 'gold' not in df.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    
    methods = [col for col in df.columns if (col not in ['sentence','gold','id'])]
    em_scores = {}
    references = [[ref] for ref in df['gold'].tolist()]
    for m in methods:
        predictions = df[m].tolist()
        em_scores[m] = bleu.compute(predictions=predictions, references=references)['score']
    return em_scores

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
    df = pd.read_csv(in_file, sep='\t')
    if 'gold' not in df.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    
    methods = [col for col in df.columns if (col not in ['sentence','gold','id'])]
    em_scores = {}
    cm = {'matrices': {}, 'labels': None}
    
    if 'skt' in in_file or 'skten' in in_file:
        labels = SKT_ENT
    elif 'lat' in in_file:
        lat_out_d = {m: 0.0 for m in methods}
        labels = LAT_ENT
        df_1 = df[df['id'].str.contains('Ovid')]
        for m in methods:
            df_1[m] = df_1.apply(lambda x: format_ner(x[m],x['sentence']), axis = 1)
            predictions = df_1[m].tolist()
            references = [ref.split() for ref in df_1['gold'].tolist()]
            scores_1 = seqeval.compute(predictions=predictions, references=references)
            #print(scores_)
            F1 = [v['f1'] for k,v in scores_1.items() if k in labels]
            lat_out_d[m] = np.mean(F1)
        print(lat_out_d)
    elif 'gra' in in_file:
        labels = GRA_ENT
    cm['labels'] = labels
    references = [ref.split() for ref in df['gold'].tolist()]
    all_refs = []
    for ref in references:
        all_refs = all_refs + [c.replace('B-','').replace('I-','') for c in ref]
    cnt_ref = Counter(all_refs)
    mc_ref = cnt_ref.most_common(6)[1:]
    mc_f1 = {m: {mr[0]: 0.0 for mr in mc_ref} for m in methods}
    mc_f1['counts'] = {mr[0]: mr[1] for mr in mc_ref}
    for m in methods:
        df[m] = df.apply(lambda x: format_ner(x[m],x['sentence']), axis = 1)
        predictions = df[m].tolist()
        all_preds = []
        for pred in predictions:
            all_preds = all_preds + [c.replace('B-','').replace('I-','') for c in pred]
        cm['matrices'][m] = confusion_matrix(all_refs, all_preds, labels=labels, normalize='true').tolist()
        scores_ = seqeval.compute(predictions=predictions, references=references)
        #print(scores_)
        F1 = [v['f1'] for k,v in scores_.items() if k in labels]
        em_scores[m] = np.mean(F1)
        for k in mc_f1[m]:
            mc_f1[m][k] = round(scores_[k]['f1'], 3)
    print(print_table_row_wise(mc_f1, ['counts']+methods, mc_f1[methods[0]].keys()))
    return em_scores, cm

category2idx = {'sanskrit': {}, 'ayurveda': {}}

skt_df = pd.read_csv("data/categories_sanskrit.tsv", sep='\t')
catgs = skt_df['category'].unique()
for c in catgs:
    id_range = skt_df[skt_df['category']==c]['id'].tolist()[0]
    lims = id_range.split('-')
    category2idx['sanskrit'][c] = list(range(int(lims[0]),int(lims[1])+1)) 

ayur_df = pd.read_csv("data/categories_ayurveda.tsv", sep='\t')
catgs = ayur_df['category'].unique()
for c in catgs:
    ids = ayur_df[ayur_df['category']==c]['id'].tolist()
    category2idx['ayurveda'][c] = ids

def eval_file_cat(in_file, dataset):
    cats = category2idx[dataset]
    df = pd.read_csv(in_file, sep='\t')
    if 'ANSWER' not in df.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    
    methods = [col for col in df.columns if (col not in ['QUESTION','ANSWER','ID','CHOICES']) and ('context' not in col) and ('maxd' not in col) and ('paths' not in col)]
    methods = ['gpt-4o','claude-3-5-sonnet-20240620','mistral-large-latest']
    em_scores = {}
    for m in methods:
        if m not in em_scores:
            em_scores[m] = {}
        for c, id_list in cats.items():
            em = df[df['ID'].isin(id_list)].apply(lambda x: compare(x['ANSWER'], x[m]), axis=1)
            em_scores[m][c] = round(em.sum()/len(em), 3)
    return em_scores

def print_table_row_wise(scores, row_labels, column_labels, row_head=''):
    lines = ['\t'.join([row_head]+[str(col) for col in column_labels])]
    for row in row_labels:
        row_scores = ['']*(len(column_labels)+1)
        line = [row]
        if row not in scores:
            continue
        row_dict = scores[row]
        for i,c in enumerate(column_labels):
            if c not in row_dict:
                row_scores[i] = '-'
            else:
                row_scores[i] = str(row_dict[c])
        line = '\t'.join(line+row_scores) 
        lines.append(line)
    return '\n'.join(lines)

def print_table_col_wise(scores, row_labels, column_labels, row_head=''):
    lines = ['\t'.join([row_head]+[str(col) for col in column_labels])]
    for row in row_labels:
        row_scores = ['']*(len(column_labels))
        line = [row]
        for i,c in enumerate(column_labels):
            if c not in scores:
                row_scores[i] = '-'
            else:
                if row in scores[c]:
                    row_scores[i] = str(scores[c][row])
                else:
                    row_scores[i] = '-'
        line = '\t'.join(line+row_scores) 
        lines.append(line)
    return '\n'.join(lines)

def zero_shot_eval(f_pth, rel_file=None, reverse=False):
    scores = {'zero-shot':{}}
    for n in range(3):
        l_f_pth = f_pth.format(n=n)
        scores_ = {}
        if os.path.exists(l_f_pth):
            if rel_file:
                scores_ = eval_file_rel(l_f_pth, rel_file,reverse=reverse)
            else:
                scores_ = eval_file(l_f_pth)
            for k,v in scores_.items():
                if k not in scores['zero-shot']:
                    scores['zero-shot'][k] = [v]
                else:
                    scores['zero-shot'][k].append(v)
    scores = {'zero-shot': {k:f"{round(np.mean(v),3)} ({round(np.std(v),3)})" for k,v in scores['zero-shot'].items()}}
    return scores['zero-shot']



def eval_default(in_file=None, ner=None, rag=None, category_wise=None, k_rag=None, zero_shot=None, rel_file=None, abl=None, mt=None):
    if in_file:
        if rel_file:
            scores = eval_file_rel(in_file, rel_file)
            print(scores)
            return

        if ner:
            scores = eval_file_ner(in_file)
            print(scores)
            return
        
        scores = eval_file(in_file)
        print(scores)
        return
    
    if abl:
        f_pth_zs = "results/zero_shot/{pre}_{{n}}.tsv"
        f_pth_rag = "results/rag/{pr}bm25_4.tsv"
        f_pth_rag_rel = "data/{pr}bm25_4_rel.tsv"
        f_pth_kg = "results/kgqa/{pre}.tsv"
        f_pth_kg_rel = "data/{pre}_kg_rel.tsv"

        for pr in ['', 'ayurveda_']:
            pre = 'sanskrit' if pr == '' else pr.replace('_','')

            ## RAG
            scores = {}
            scores['zero-shot'] = zero_shot_eval(f_pth_zs.format(pre=pre), f_pth_rag_rel.format(pr=pr))
            scores['rag-bm25'] = eval_file_rel(f_pth_rag.format(pr=pr), f_pth_rag_rel.format(pr=pr))

            res_txt = print_table_row_wise(scores, ['zero-shot', 'rag-bm25'], DEFAULT_MODELS, row_head='Method')
            print(f"Dataset: {'ramayana' if pr == '' else pre} RAG")
            print(res_txt)
            print()
            with open(f"results/rag/eval_table_{pr}rag_abl.tsv",'w') as fp:
                fp.write(res_txt)

            ## RAG inverse
            scores = {}
            scores['zero-shot'] = zero_shot_eval(f_pth_zs.format(pre=pre), f_pth_rag_rel.format(pr=pr), reverse=True)
            scores['rag-bm25'] = eval_file_rel(f_pth_rag.format(pr=pr), f_pth_rag_rel.format(pr=pr), reverse=True)

            res_txt = print_table_row_wise(scores, ['zero-shot', 'rag-bm25'], DEFAULT_MODELS, row_head='Method')
            print(f"Dataset: {'ramayana' if pr == '' else pre} RAG inverse")
            print(res_txt)
            print()
            with open(f"results/rag/eval_table_{pr}rag_abl_inverse.tsv",'w') as fp:
                fp.write(res_txt)

            ## KG
            scores = {}
            scores['zero-shot'] = zero_shot_eval(f_pth_zs.format(pre=pre), f_pth_kg_rel.format(pre=pre))
            scores['rag-bm25'] = eval_file_rel(f_pth_rag.format(pr=pr), f_pth_kg_rel.format(pre=pre))
            scores['llm-kg'] = eval_file_rel(f_pth_kg.format(pre=pre), f_pth_kg_rel.format(pre=pre))

            res_txt = print_table_row_wise(scores, ['zero-shot', 'rag-bm25', 'llm-kg'], DEFAULT_MODELS, row_head='Method')
            print(f"Dataset: {'ramayana' if pr == '' else pre} LLM-KG")
            print(res_txt)
            print()
            with open(f"results/kgqa/eval_table_{pre}_abl.tsv",'w') as fp:
                fp.write(res_txt)

            ## KG inverse
            scores = {}
            scores['zero-shot'] = zero_shot_eval(f_pth_zs.format(pre=pre), f_pth_kg_rel.format(pre=pre), reverse=True)
            scores['rag-bm25'] = eval_file_rel(f_pth_rag.format(pr=pr), f_pth_kg_rel.format(pre=pre), reverse=True)
            scores['llm-kg'] = eval_file_rel(f_pth_kg.format(pre=pre), f_pth_kg_rel.format(pre=pre), reverse=True)


            res_txt = print_table_row_wise(scores, ['zero-shot', 'rag-bm25', 'llm-kg'], DEFAULT_MODELS, row_head='Method')
            print(f"Dataset: {'ramayana' if pr == '' else pre} LLM-KG inverse")
            print(res_txt)
            print()
            with open(f"results/kgqa/eval_table_{pre}_abl_inverse.tsv",'w') as fp:
                fp.write(res_txt)




    if zero_shot:
        f_pth = "results/zero_shot/{lang}_{n}.tsv"
        lang = ['sanskrit','ayurveda']
        scores = {}
        methods = set()
        for l in lang:
            for n in range(3):
                l_f_pth = f_pth.format(lang=l, n=n)
                scores_ = {}
                if os.path.exists(l_f_pth):
                    if l not in scores:
                        scores[l] = {}
                    scores_ = eval_file(l_f_pth)
                    methods = methods.union(list(scores[l].keys()))
                    for k,v in scores_.items():
                        if k not in scores[l]:
                            scores[l][k] = [v]
                        else:
                            scores[l][k].append(v)
        scores_w_bars = {l: {k:f"{round(np.mean(v),3)} ({round(np.std(v),3)})" for k,v in d.items()} for l,d in scores.items()}
        res_txt = print_table_col_wise(scores_w_bars, DEFAULT_MODELS+LOW_END_MODELS, lang, row_head='LLM')
        print(res_txt)
        with open("results/zero_shot/eval_table.tsv",'w') as fp:
            fp.write(res_txt)   

    if ner:
        f_pth = "results/ner/{lang}_{n}.tsv"
        lang = ['skten_ner', 'lat_ner', 'gra_ner']
        scores = {}
        cm = {}
        methods = set()
        for l in lang:
            for n in range(1):
                l_f_pth = f_pth.format(lang=l, n=n)
                scores_ = {}
                if os.path.exists(l_f_pth):
                    if l not in scores:
                        scores[l] = {}
                    scores_, cm[l] = eval_file_ner(l_f_pth)
                    methods = methods.union(list(scores[l].keys()))
                    for k,v in scores_.items():
                        if k not in scores[l]:
                            scores[l][k] = [v]
                        else:
                            scores[l][k].append(v)
        scores_w_bars = {l: {k:f"{round(np.mean(v),3)} ({round(np.std(v),3)})" for k,v in d.items()} for l,d in scores.items()}
        res_txt = print_table_col_wise(scores_w_bars, DEFAULT_MODELS+LOW_END_MODELS, lang, row_head='LLM')
        print(res_txt)
        with open("results/ner/ner_confusion.json",'w') as fp:
            json.dump(cm,fp, indent='\t')
        with open("results/ner/eval_table.tsv",'w') as fp:
            fp.write(res_txt) 
    if mt:
        f_pth = "results/mt/{lang}_{n}.tsv"
        lang = ['mt_in','grc_eng','lat_eng']#, 'mt_out'
        scores = {}
        methods = set()
        for l in lang:
            for n in range(1):
                l_f_pth = f_pth.format(lang=l, n=n)
                scores_ = {}
                if os.path.exists(l_f_pth):
                    if l not in scores:
                        scores[l] = {}
                    scores_ = eval_file_mt(l_f_pth)
                    methods = methods.union(list(scores[l].keys()))
                    for k,v in scores_.items():
                        if k not in scores[l]:
                            scores[l][k] = [v]
                        else:
                            scores[l][k].append(v)
        scores_w_bars = {l: {k:f"{round(np.mean(v),3)} ({round(np.std(v),3)})" for k,v in d.items()} for l,d in scores.items()}
        res_txt = print_table_col_wise(scores_w_bars, DEFAULT_MODELS+LOW_END_MODELS, lang, row_head='LLM')
        print(res_txt)
        with open("results/mt/eval_table.tsv",'w') as fp:
            fp.write(res_txt) 
    if rag:
        f_pth = "results/rag/{pre}{embedding}_4.tsv"
        f_pth_kg = "results/kgqa/{pre}.tsv"
        f_pth_zs = "results/zero_shot/eval_table.tsv"
        emb = ['bm25', 'fasttext','glove']
        for pr in ['','ayurveda_']:
            scores = {}
            for e in emb:
                e_f_pth = f_pth.format(embedding=e, pre=pr)
                if os.path.exists(e_f_pth):
                    scores[e] = eval_file(e_f_pth)

            dataset = 'sanskrit' if pr == '' else pr.replace('_','')
            kg_f_pth = f_pth_kg.format(pre = dataset)
            
            if os.path.exists(kg_f_pth):
                scores['llm-kg'] = eval_file(kg_f_pth)
            
            if os.path.exists(f_pth_zs):
                zs_df = pd.read_csv(f_pth_zs, sep='\t')
                scores['zero-shot'] = dict(zip(zs_df['LLM'], zs_df[dataset]))


            res_txt = print_table_row_wise(scores, ['zero-shot']+emb+['llm-kg'], DEFAULT_MODELS+LOW_END_MODELS, row_head='Method')
            print(f"Dataset: {'ramayana' if pr == '' else dataset}")
            print(res_txt)
            print()
            with open(f"results/rag/eval_table_{pr}k4.tsv",'w') as fp:
                fp.write(res_txt)

    if category_wise:
        f_pth = "results/rag/{pre}{embedding}_4.tsv"
        f_pth_kg = "results/kgqa/{pre}.tsv"
        f_pth_zs = "results/zero_shot/{pre}_0.tsv"

        emb = ['bm25'] #, 'fasttext','glove']
        for pr in ['','ayurveda_']:

            dataset = 'sanskrit' if pr == '' else pr.replace('_','')
            scores = {}
            for e in emb:
                e_f_pth = f_pth.format(embedding=e, pre=pr)
                if os.path.exists(e_f_pth):
                    scores[e] = eval_file_cat(e_f_pth, dataset)

            kg_f_pth = f_pth_kg.format(pre = dataset)
            zs_f_pth = f_pth_zs.format(pre = dataset)
            
            if os.path.exists(kg_f_pth):
                scores['llm-kg'] = eval_file_cat(kg_f_pth, dataset)
            
            if os.path.exists(zs_f_pth):
                scores['zero-shot'] = eval_file_cat(zs_f_pth, dataset)

            new_scores = {}
            methods = list(scores['zero-shot'].keys())
            paradigms = ['zero-shot'] + emb + ['llm-kg']

            for p in paradigms:
                for m in methods:
                    new_scores[f"{p}-{m}"] = scores[p][m]
            res_txt = print_table_row_wise(new_scores, list(new_scores.keys()), list(new_scores[f"{paradigms[0]}-{methods[0]}"].keys()), row_head='Model')
            print(f"Dataset: {'ramayana' if pr == '' else dataset}")
            print(res_txt)
            print()
            with open(f"results/category_wise/eval_table_{pr}k4.tsv",'w') as fp:
                fp.write(res_txt)

    if k_rag:
        f_pth = "results/rag/{pre}{embedding}_{k}.tsv"
        emb = ['bm25', 'fasttext', 'glove']
        for pr in ['', 'ayurveda_']:
            scores = {}
            for e in emb:
                scores[e] = {}
                for k in range(1,MAX_K+1):
                    e_f_pth = f_pth.format(pre=pr, embedding=e, k=k)
                    if os.path.exists(e_f_pth):
                        scores[e][k] = eval_file(e_f_pth)['gpt-4o']
            print(f"Dataset: {pr} (default Ramayana)")
            res_txt = print_table_row_wise(scores, emb, list(range(1,MAX_K+1)), row_head='Retriever')
            print(res_txt)
            plot_k(scores, pre=pr)
            with open(f"results/rag/eval_table_{pr}gpt4o.tsv",'w') as fp:
                fp.write(res_txt)


        



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate a subtask given the files follow default conventions or just evaluate a single file")
    parser.add_argument('-i','--in-file',type=str,help="input tsv file to evaluate")
    parser.add_argument('-l','--rel-file',type=str,help="relavence tsv file to compare")
    parser.add_argument('-z','--zero-shot', action='store_true', help="evaluate zero-shot QA")
    parser.add_argument('-t','--mt',action='store_true', help="evaluate machine translation")
    parser.add_argument('-n','--ner',action='store_true', help="evaluate NER")
    parser.add_argument('-r','--rag',action='store_true', help="evaluate RAG for k=4 across available methods")
    parser.add_argument('-c','--category-wise',action='store_true', help="evaluate category wise")
    parser.add_argument('-a', '--abl', action='store_true', help="generate ablation results")
    parser.add_argument('-k','--k-rag',action='store_true', help="evaluate RAG for GPT-4o across k=1,2,3,4")
    args = parser.parse_args()

    args_dict = vars(args)
    eval_default(**args_dict)