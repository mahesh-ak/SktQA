import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt


def plot_k(data):
    for key, values in data.items():
        plt.plot(values.keys(), values.values(), label=key, marker='o')

    # Labels and title
    plt.xlabel('k')
    plt.ylabel('EM score')
    plt.xticks(range(1,4))
    plt.title('Effect of k value in RAG for GPT-4o')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.savefig('results/rag/rag_k_plot_gpt4o.png')

def eval_file(in_file):
    df = pd.read_csv(in_file, sep='\t')
    if 'ANSWER' not in df.columns:
        print('Error: gold answers should be present in column ANSWER')
        exit(1)
    
    methods = [col for col in df.columns if (col not in ['QUESTION','ANSWER']) and ('context' not in col)]
    em_scores = {}
    for m in methods:
        em = df.apply(lambda x: str(x['ANSWER']).strip() == str(x[m]).strip(), axis=1)
        em_scores[m] = round(em.sum()/len(em), 3)
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
        row_scores = ['']*(len(column_labels)+1)
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

def eval_default(in_file=None, rag=None, k_rag=None, zero_shot=None):
    if in_file:
        scores = eval_file(in_file)
        print(scores)
        return
    
    if zero_shot:
        f_pth = "results/zero_shot/{lang}.tsv"
        lang = ['sanskrit','telugu','hindi','bengali','marathi']
        scores = {}
        methods = set()
        for l in lang:
            l_f_pth = f_pth.format(lang=l)
            if os.path.exists(l_f_pth):
                scores[l] = eval_file(l_f_pth)
                methods = methods.union(list(scores[l].keys()))
        res_txt = print_table_col_wise(scores, list(methods), lang, row_head='LLM')
        print(res_txt)
        with open("results/zero_shot/eval_table.tsv",'w') as fp:
            fp.write(res_txt)   

    if rag:
        f_pth = "results/rag/{embedding}_2.tsv"
        emb = ['bm25', 'fasttext', 'glove']
        scores = {}
        methods = set()
        for e in emb:
            e_f_pth = f_pth.format(embedding=e)
            if os.path.exists(e_f_pth):
                scores[e] = eval_file(e_f_pth)
                methods = methods.union(list(scores[e].keys()))
        res_txt = print_table_row_wise(scores, emb, list(methods), row_head='Retriever')
        print(res_txt)
        with open("results/rag/eval_table_k2.tsv",'w') as fp:
            fp.write(res_txt)
    
    if k_rag:
        f_pth = "results/rag/{embedding}_{k}.tsv"
        emb = ['bm25', 'fasttext', 'glove']
        scores = {}
        for e in emb:
            scores[e] = {}
            for k in range(1,4):
                e_f_pth = f_pth.format(embedding=e, k=k)
                if os.path.exists(e_f_pth):
                    scores[e][k] = eval_file(e_f_pth)['gpt-4o']
        res_txt = print_table_row_wise(scores, emb, list(range(1,4)), row_head='Retriever')
        print(res_txt)
        plot_k(scores)
        with open("results/rag/eval_table_gpt4o.tsv",'w') as fp:
            fp.write(res_txt)


        



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate a subtask given the files follow default conventions or just evaluate a single file")
    parser.add_argument('-i','--in-file',type=str,help="input tsv file to evaluate")
    parser.add_argument('-z','--zero-shot', action='store_true', help="evaluate zero-shot QA")
    parser.add_argument('-r','--rag',action='store_true', help="evaluate RAG for k=2 across available methods")
    parser.add_argument('-k','--k-rag',action='store_true', help="evaluate RAG for GPT-4o across k=1,2,3")
    args = parser.parse_args()

    args_dict = vars(args)
    eval_default(**args_dict)