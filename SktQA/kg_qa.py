import argparse
import os
from utils import *

MAX_DEPTH = 3
MAX_WIDTH = 3

def ToG(question, entity_chain, traverse_chain):
    pass

def run_kgqa(in_file, model, out_file=None, force=None):
    if out_file==None:
        out_pth = "results/kgqa"
        os.makedirs(out_pth, exist_ok=True)
        out_file = os.path.join(out_pth, f"sanskrit.tsv")
    
    in_df = pd.read_csv(in_file, sep='\t')
    if os.path.exists(out_file):
        out_df = pd.read_csv(out_file, sep='\t')
    else:
        out_df = in_df.copy()
    
    entity_chain = None # Put
    traverse_chain = None # Put
    
    if model in out_df.columns and (not force):
        print(f"Warning! column {model} already exists in {out_file}. Skipping the chain. Specify -f to overwrite.")
    else:
        print(f"Applying ToG for KGQA on {in_file} with {model}, saving to {out_file}")
        in_df[model] = in_df.apply(lambda x: ToG(x['QUESTION']), axis=1)
        out_df[model] = in_df.apply(lambda x: x[model]['answer'], axis=1)


    out_df.to_csv(out_file, sep='\t', index=False)

    return


def run_kgqa_default(in_file, model= None, **kwargs):
    if model == None:
        models = DEFAULT_MODELS
    else:
        models = [model]

    for m in models:
        run_kgqa(in_file, m, **kwargs)
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run KGQA with LLM Agent by Think-on-Graph (ToG)")
    parser.add_argument('-i','--in-file',type=str, default= 'data/qa_set/sanskrit.tsv',help="Path to input tsv file with columns QUESTION, ANSWER")
#    parser.add_argument('-l','--lang', type=str, help="Language of input file, by default deduces from [IN] assuming format [LANG].tsv")
    parser.add_argument('-m','--model',type=str, help= "LLM model name, currently supports: gpt-*, claude-*, gemini-*, mistral-*, llama-*, by default runs on gpt-4o, claude-3-5-sonnet, gemini-1.5-pro, mistral-large and llama-v3p1-8b-instruct")
    parser.add_argument('-o','--out-file',type=str, help="out file name to store predictions, by default stores in results/kgqa/[LANG].tsv")
    parser.add_argument('-f','--force', action='store_true', help="overwrite the outfile column")
    args = parser.parse_args()

    args_dict = vars(args)
    run_kgqa_default(**args_dict)

