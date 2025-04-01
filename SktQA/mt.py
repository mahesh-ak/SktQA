from utils import *
import argparse
from indic_transliteration.sanscript import IAST, DEVANAGARI, transliterate
from langchain_core.prompts.chat import ChatPromptTemplate

def MTChain(model='gpt-4o', language='english'):
    
    match language:
        case 'mt_in' | 'mt_out':
            template = f"अधो दत्त-संस्कृत-वाक्यम्‌ आंग्ले अनुवादय, तद्‌ अपि विवृतम्‌ मा कुरु -"
        case 'mt_in_en' | 'mt_out_en' | 'mt_in_iast' | 'mt_out_iast':
            template = f"Translate the following sentence in Sanskrit into English. Do not give any explanations."
        case 'grc_eng':
            template = f"Translate the following sentence in Ancient Greek into English. Do not give any explanations."
        case 'grc_eng_nen':
            template = f"Μετάφρασον τὴνδε τὴν Ἑλληνικὴν πρότασιν εἰς τὴν Ἀγγλικήν. Μηδεμίαν ἐξήγησιν παρέχου."
        case 'lat_eng':
            template = f"Translate the following sentence in Latin into English. Do not give any explanations."
        case 'lat_eng_nen':
            template = f"Verte hanc sententiam Latinam in Anglicam. Nullam explicationem praebe."

    
    human_template = "{input}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    def output_parse(ai_message):
        return ai_message.content.strip()

    chat_model = get_chat_model(model=model)    
    mt_chain = chat_prompt | chat_model | output_parse

    return mt_chain

def run_mt(in_file, model, lang=None, out_file=None, force=None, r=None):
    if lang==None:
        lang = os.path.split(in_file)[-1].replace(".tsv","")
    if out_file==None:
        out_pth = "results/mt"
        out_fname = f"{lang}_{0}.tsv"
        if r:
            out_fname = f"{lang}_{r}.tsv"
        os.makedirs(out_pth, exist_ok=True)
        out_file = os.path.join(out_pth, out_fname)

    in_df = pd.read_csv(in_file.replace('_en.tsv','.tsv').replace('_nen.tsv','.tsv').replace('_iast.tsv','.tsv'), sep='\t')
#    in_df = in_df
    if 'iast' in in_file:
        in_df['sentence'] = in_df.apply(lambda x: transliterate(x['sentence'], DEVANAGARI, IAST), axis=1)
    if os.path.exists(out_file):
        out_df = pd.read_csv(out_file, sep='\t')
    else:
        out_df = in_df
    
    chain = MTChain(model=model, language=lang)
    if model in out_df.columns and (not force):
        print(f"Warning! column {model} already exists in {out_file}. Skipping the chain. Specify -f to overwrite.")
    else:
        print(f"Applying MT chain on {in_file} with {model}, saving to {out_file}")
        in_df = ApplyChainOnDF(in_df, chain=chain, col_name= model)
        out_df[model] = in_df[model]

    out_df.to_csv(out_file, sep='\t', index=False)
    return

def run_mt_default(in_file=None, model=None,repeat=None, **kwargs):
    if in_file == None:
        file_path = 'data/mt/'
        f_pths = [os.path.join(file_path, f_name) for f_name in ['mt_in.tsv','mt_out.tsv','grc_eng.tsv','lat_eng.tsv', 'grc_eng_nen.tsv', 'lat_eng_nen.tsv', 'mt_in_en.tsv', 'mt_out_en.tsv','mt_in_iast.tsv', 'mt_out_iast.tsv']]
    else:
        f_pths = [in_file]
    
    if model == None:
        models = DEFAULT_MODELS + LOW_END_MODELS
    else:
        models = [model]
    
    for fl in f_pths:
        for m in models:
            if repeat:
                for n in range(3):
                    run_mt(fl,m, r=n,**kwargs)
            else:
                run_mt(fl, m, **kwargs)
    return

    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run MT evaluation using LLMs")
    parser.add_argument('-i','--in-file',type=str, help="input file containing columns 'sentence', 'gold'")
    parser.add_argument('-l','--lang', type=str, help="Language of input file")
    parser.add_argument('-m','--model',type=str, help= "LLM model name, currently supports: gpt-*, claude-*, gemini-*, mistral-*, llama-*, by default runs on gpt-4o*, llama-v3p1-*b-instruct")
    parser.add_argument('-o','--out-file',type=str, help="out file name to store predictions, by default stores in results/mt/[LANG].tsv")
    parser.add_argument('-f','--force', action='store_true', help="overwrite the outfile column")
    parser.add_argument('-r','--repeat',action='store_true',help="Repeat experiment 3 times")
    args = parser.parse_args()

    args_dict = vars(args)
    run_mt_default(**args_dict)
