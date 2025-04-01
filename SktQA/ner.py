from utils import *
import argparse
from langchain_core.prompts.chat import ChatPromptTemplate
from indic_transliteration.sanscript import IAST, DEVANAGARI, transliterate

def NERChain(model='gpt-4o', language='english'):
    
    match language:
        case 'skten_ner' | 'sktiast_ner':
            template = """Recognize the named entities from the following sentence in Sanskrit.
The valid entities are 'O', 'ASURA', 'RAKSHASA', 'HUMAN', 'KULA', 'DEVA', 'PALACE', 'NAGA', 'GANDHARVA', 'TREE', 'FLOWER', 'MOUNTAIN', 'KINGDOM', 'VANARA', 'AXE', 'ORNAMENT', 'MUHURTA', 'SEA', 'HOUSE', 'GARDEN', 'FOREST', 'ASTRA', 'VINE', 'RIVERBANK', 'GRAHA', 'CITY', 'GRIDHRA', 'ARROW', 'ROAD', 'FESTIVAL', 'SWARGA', 'FRUIT', 'RATHA' । 
Do not provide explanation and do not list out entries of 'O'. Example:
Sentence: <word_1> <word_2> <word_3> <word_4> <word_5>
Output: {{'B-<entity1>': ['<word_1>', '<word_4>'], 'B-<entity2>':['<word_5>']}}"""
            human_template = """Sentence: {input}
Output: """

        case 'skt_ner':
            template = """अधो दत्त वाक्ये नामकृताः सत्त्वाः (named entities) अभिजानीहि । तदपि विवृतम्‌ मा कुरु, केवलम्‌ पृष्ट-विषयस्य उत्तरम्‌ देहि । अपि च 'O'-सम्बन्धितानि न देयानि । 
सत्त्वाः एतेषु वर्गेषु वर्तन्ते - 'O', 'ASURA', 'RAKSHASA', 'HUMAN', 'KULA', 'DEVA', 'PALACE', 'NAGA', 'GANDHARVA', 'TREE', 'FLOWER', 'MOUNTAIN', 'KINGDOM', 'VANARA', 'AXE', 'ORNAMENT', 'MUHURTA', 'SEA', 'HOUSE', 'GARDEN', 'FOREST', 'ASTRA', 'VINE', 'RIVERBANK', 'GRAHA', 'CITY', 'GRIDHRA', 'ARROW', 'ROAD', 'FESTIVAL', 'SWARGA', 'FRUIT', 'RATHA' । 
उदाहरणाय - 
वाक्यम्‌: <पदम्‌_1> <पदम्‌_2> <पदम्‌_3> <पदम्‌_4> <पदम्‌_5>
फलितम्‌: {{'B-<सत्त्व:1>': ['<पदम्‌_1>', '<पदम्‌_4>'], 'B-<सत्त्व:2>': ['<पदम्‌_5>']}}"""
            human_template = """वाक्यम्‌: {input}
फलितम्‌: """

        case 'lat_ner':
            template = """Recognize the named entities from the following Latin sentence.
The valid entities 'O', 'PERS', 'LOC', 'GRP'.
Do not provide explanation and do not list out entries of 'O'. Example:
Sentence: <word_1> <word_2> <word_3> <word_4> <word_5>
Output: {{'B-<entity1>': ['<word_1>', '<word_4>'], 'I-<entity1>': ['<word_2>'], 'B-<entity2>':['<word_5>']}}"""
            human_template = """Sentence: {input}
Output: """

        case 'lat_nenner':
            template = """Agnosce nomina propria (named entities) ex hac sententia Latina.
Notae validae sunt 'O', 'PERS', 'LOC', 'GRP'.
Explanationem ne praebeas nec elementa 'O' elenca. Exemplar:
Sententia: <verbum_1> <verbum_2> <verbum_3> <verbum_4> <verbum_5>
Productus: {{'B-<entitatem1>': ['<verbum_1>', '<verbum_4>'], 'I-<entitatem1>': ['<verbum_2>'], 'B-<entitatem3>':['<verbum_5>']}}"""
            human_template = """Sententia: {input}
Productus: """

        case 'gra_ner':
            template = """Recognize the named entities from the following sentence in Ancient Greek.
The valid entities are 'O', 'LOC', 'GOD', 'ORG', 'NORP', 'WORK', 'EVENT', 'PERSON', 'LANGUAGE'.
NORP refers to ethnic groups (like greeks, persians), demonyms and other social groups (like religious organizations)
Do not provide explanation in the answer and do not list out entries of 'O'. Example:
Sentence: <word_1> <word_2> <word_3> <word_4> <word_5>
Output: {{'B-<entity1>': ['<word_1>', '<word_4>'], 'B-<entity2>':['<word_5>']}}""" 
            human_template = """Sentence: {input}
Output: """

        case 'gra_nenner':
            template = """Ἀναγνώρισον τὰ ὀνόματα (named entities) ἐκ τῆςδε τῆς Ἑλληνικῆς περιόδου.
Τὰ ἔγκυρα εἴδη ἐστιν 'O', 'LOC', 'GOD', 'ORG', 'NORP', 'WORK', 'EVENT', 'PERSON', 'LANGUAGE'.
NORP σημαίνει ἔθνη (οἷον Ἕλληνες, Πέρσαι), ἐθνωνύμια, καὶ ἄλλας κοινωνικὰς ὁμάδας (οἷον θρησκευτικὰς ὁργανώσεις).
Μὴ παρέχου ἐξήγησιν ἐν τῇ ἀποκρίσει μηδὲ τὰ εἰς 'O' ἐγγεγραμμένα παρατίθεσο. Παράδειγμα:
Πρότασις: <λέξις_1> <λέξις_2> <λέξις_3> <λέξις_4> <λέξις_5>
Παραγωγή: {{'B-<ὀντότης1>': ['<λέξις_1>', '<λέξις_4>'], 'B-<ὀντότης2>':['<λέξις_5>']}}"""
            human_template = """Πρότασις: {input}
Παραγωγή: """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    output_tags = ['Output:', 'Productus:','Παραγωγή:', 'फलितम्‌:']
    def output_parse(ai_message):
        output_txt = ai_message.content
        for tag in output_tags:
            if tag in output_txt:
                output_txt = output_txt.split(tag)[1]
        return output_txt.strip()

    chat_model = get_chat_model(model=model)    
    ner_chain = chat_prompt | chat_model | output_parse

    return ner_chain

def run_ner(in_file, model, lang=None, out_file=None, force=None, r=None):
    if lang==None:
        lang = os.path.split(in_file)[-1].replace(".tsv","")
    if out_file==None:
        out_pth = "results/ner"
        out_fname = f"{lang}_{0}.tsv"
        if r:
            out_fname = f"{lang}_{r}.tsv"
        os.makedirs(out_pth, exist_ok=True)
        out_file = os.path.join(out_pth, out_fname)

    in_file = in_file.replace("skten","skt").replace("nen","").replace("iast","")
    in_df = pd.read_csv(in_file, sep='\t')
    #in_df = in_df[:100]
    if 'iast' in lang:
        in_df['sentence'] = in_df.apply(lambda x: transliterate(x['sentence'], DEVANAGARI, IAST), axis=1)
    if os.path.exists(out_file):
        out_df = pd.read_csv(out_file, sep='\t')
    else:
        out_df = in_df
    
    chain = NERChain(model=model, language=lang)
    if model in out_df.columns and (not force):
        print(f"Warning! column {model} already exists in {out_file}. Skipping the chain. Specify -f to overwrite.")
    else:
        print(f"Applying NER chain on {in_file} with {model}, saving to {out_file}")
        in_df = ApplyChainOnDF(in_df, chain=chain, col_name= model)
        out_df[model] = in_df[model]

    out_df.to_csv(out_file, sep='\t', index=False)
    return

def run_ner_default(in_file=None, model=None,repeat=None, **kwargs):
    if in_file == None:
        file_path = 'data/ner/'
        f_pths = [os.path.join(file_path, f_name) for f_name in ['skten_ner.tsv','skt_ner.tsv','lat_ner.tsv', 'gra_ner.tsv','lat_nenner.tsv','gra_nenner.tsv', 'sktiast_ner.tsv']]
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
                    run_ner(fl,m, r=n,**kwargs)
            else:
                run_ner(fl, m, **kwargs)
    return

    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run NER evaluation using LLMs")
    parser.add_argument('-i','--in-file',type=str, help="input file containing columns 'sentence', 'gold'")
    parser.add_argument('-l','--lang', type=str, help="Language of input file")
    parser.add_argument('-m','--model',type=str, help= "LLM model name, currently supports: gpt-*, claude-*, gemini-*, mistral-*, llama-*, by default runs on gpt-4o*, llama-v3p1-*b-instruct")
    parser.add_argument('-o','--out-file',type=str, help="out file name to store predictions, by default stores in results/ner/[LANG].tsv")
    parser.add_argument('-f','--force', action='store_true', help="overwrite the outfile column")
    parser.add_argument('-r','--repeat',action='store_true',help="Repeat experiment 3 times")
    args = parser.parse_args()

    args_dict = vars(args)
    run_ner_default(**args_dict)
