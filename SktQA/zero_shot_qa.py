from utils import *
import argparse
from langchain_core.prompts.chat import ChatPromptTemplate

def ZeroShotChain(model='gpt-4o', dataset='sanskrit', language='english'):
    
    text = 'आयुर्वेद' if dataset.replace('_en','') == 'ayurveda' else 'रामायण'

    match language:
        case 'sanskrit':
            template = f"त्वया संस्कृत-भाषायाम् एव वक्तव्यम्। न तु अन्यासु भाषासु। अधः {text}-सम्बन्धे पृष्ट-प्रश्नस्य प्रत्युत्तरं देहि। तदपि एकेनैव पदेन यदि उत्तरे कारणं नापेक्षितम्। कथम् किमर्थम् इत्यादिषु एकेन लघु वाक्येन उत्तरं देहि अत्र तु एक-पद-नियमः नास्ति। "
        case 'english'| _:
            print("Warning! Unspecified language, defaulting prompt to English")
            template = f"Answer the question related to {text} in the Sanskrit only. Give a single word answer if reasoning is not demanded in the answer. With regards to how-questions, answer in a short phrase, there is no single word restriction."
    
    
    human_template = "{question} {choices}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    def output_parse(ai_message):
        return ai_message.content.replace('।','').replace('.','').strip()

    chat_model = get_chat_model(model=model)    
    zeroshot_chain = chat_prompt | chat_model | output_parse

    return zeroshot_chain

def run_zero_shot_qa(in_file, model, lang=None, out_file=None, force=None, r=None):
    if lang==None:
        lang = os.path.split(in_file)[-1].replace(".tsv","")
    if out_file==None:
        out_pth = "results/zero_shot"
        out_fname = f"{lang}_{0}.tsv"
        if r:
            out_fname = f"{lang}_{r}.tsv"
        os.makedirs(out_pth, exist_ok=True)
        out_file = os.path.join(out_pth, out_fname)
    dataset = lang
    lang = lang.replace("ayurveda","sanskrit")

    in_df = pd.read_csv(in_file.replace("_en",""), sep='\t')
    if os.path.exists(out_file):
        out_df = pd.read_csv(out_file, sep='\t')
    else:
        out_df = in_df
    
    out_df['ANSWER'] = in_df['ANSWER']

    chain = ZeroShotChain(model=model, dataset=dataset, language=lang)
    if model in out_df.columns and (not force):
        print(f"Warning! column {model} already exists in {out_file}. Skipping the chain. Specify -f to overwrite.")
    else:
        print(f"Applying zero-shot chain on {in_file} with {model}, saving to {out_file}")
        in_df = ApplyQAChainOnDF(in_df, chain=chain, col_name= model)
        out_df[model] = in_df[model]

    out_df.to_csv(out_file, sep='\t', index=False)
    return

def run_zero_shot_default(in_file=None, model=None,repeat=None, **kwargs):
    if in_file == None:
        file_path = 'data/qa_set'
        f_pths = [os.path.join(file_path, f_name) for f_name in ['sanskrit.tsv', 'ayurveda.tsv', 'sanskrit_en.tsv', 'ayurveda_en.tsv']]
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
                    run_zero_shot_qa(fl,m, r=n,**kwargs)
            else:
                run_zero_shot_qa(fl, m, **kwargs)
    return

    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run zero-shot QA evaluation using LLMs")
    parser.add_argument('-i','--in-file',type=str, help="input file containing columns 'QUESTION', 'ANSWER' (gold). By default runs on all languages")
    parser.add_argument('-l','--lang', type=str, help="Language of input file, by default deduces from [IN] assuming format [LANG].tsv")
    parser.add_argument('-m','--model',type=str, help= "LLM model name, currently supports: gpt-*, claude-*, gemini-*, mistral-*, llama-*, by default runs on gpt-4o, claude-3-5-sonnet, gemini-1.5-pro, mistral-large and llama-v3p1-8b-instruct")
    parser.add_argument('-o','--out-file',type=str, help="out file name to store predictions, by default stores in results/zero_shot/[LANG].tsv")
    parser.add_argument('-f','--force', action='store_true', help="overwrite the outfile column")
    parser.add_argument('-r','--repeat',action='store_true',help="Repeat experiment 3 times")
    args = parser.parse_args()

    args_dict = vars(args)
    run_zero_shot_default(**args_dict)
