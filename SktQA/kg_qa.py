from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import argparse
import os
from utils import *
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
from indic_transliteration.sanscript import IAST, DEVANAGARI, transliterate
import torch
import regex as re
import random

MAX_DEPTH = 2
MAX_WIDTH = 3
MAX_WID_PER_ELEM = 15

graph = Neo4jGraph()

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the named entities appearing in the text",
    )
    scores: List[float] = Field(
        ...,
        description="Relevance scores of the named entities appearing in the text",
    )

    paths: List[List[str]] = Field(
        ...,
        description="History paths of the entities"
    )

    def append_paths(self, track_paths):
        new_names = []
        new_scores = []
        new_paths = []
    
        for e, s in zip(self.names,self.scores):
            lemma = e.strip()
            if lemma in track_paths:
                new_names.append(lemma)
                new_scores.append(s)
                new_paths.append(track_paths[lemma])
        self.names = new_names
        self.scores = new_scores
        self.paths = new_paths


class Relationships(BaseModel):
    """Identifying information about Relationships."""

    names: List[str] = Field(
        ...,
        description="All the relationships listed"
    ) 
    
    
    scores: List[float] = Field(
        ...,
        description="Relevance scores of the relationships",
    )


class Lemmatizer:
    checkpoint = 'mahesh27/t5lemmatizer'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    lemmatizer_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    def lemmatize(self, chunk, translate_only=False):
        text = chunk.replace('\n\n','\n').split('\n')
        pattern = re.compile(r'[0-9]+,[0-9]+\|[0-9]+')
        lines = [pattern.sub('',transliterate(line, DEVANAGARI, IAST)) for line in text]
        if translate_only:
            return ' '.join(lines)
        tokenized_text = [{'input_ids': self.tokenizer(line)['input_ids'] + [self.tokenizer.eos_token_id]} for line in lines]
    
        inputs = self.data_collator(tokenized_text)
        with torch.no_grad():
            outputs = self.lemmatizer_model.generate(inputs=inputs['input_ids'], max_length=64)
    
        return '\n'.join(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))

    def entity_lemmas(self, entities: Entities, lemmatized=False):
        length = len(entities.names)
        entities.names = [''.join(x.split()) for x in entities.names]

        if not lemmatized:
            lemmatized_names = transliterate(corr_diatrics(self.lemmatize(' '.join(entities.names)).replace('\n',' ')), IAST, DEVANAGARI).split()
            if len(lemmatized_names) != length:
                lemmatized = self.lemmatize('\n'.join(entities.names)).split('\n')
                lemmatized_names = transliterate(corr_diatrics(' '.join([''.join(x.split()) for x in lemmatized])), IAST, DEVANAGARI).split()

            entities.names = lemmatized_names
        entities.paths = [['']]*length
        return entities

def chain_invoke_(chain, inp):
    done = False
    ret = None
    i = 0
    flag = 0
    while not done:
        try:
            done = True
            ret =  chain.invoke(inp)
        except ValidationError:
            done = True
            ret = None
        except ValueError:
            done = True
            ret = None
        except Exception as e:
            if i > 128:
                ret = None
                done = True
                break
            done = False
            print(e)
            i += 1
            print(f"Reattempting {i}th time in a row...")

    if ret is None:
        flag = 1
    return ret, flag


class ToG:

    @staticmethod
    def prune(items, th=MAX_WIDTH):
        if items == None:
            return items
        
        size = len(items.names)

        if size == 0:
            return items
        size = min(size, len(items.scores))
        lim = min(size, th)
        names_tuple = [(items.names[i], items.scores[i]) for i in range(size)]
        names_tuple.sort(key=lambda x: x[1], reverse=True)
        items.names = [names_tuple[i][0] for i in range(lim)]
    
        if 'paths' in vars(items):
            paths_tuple = [(items.paths[i], items.scores[i]) for i in range(size)]
            paths_tuple.sort(key=lambda x: x[1], reverse=True)
            items.paths = [paths_tuple[i][0] for i in range(lim)]

        items.scores = [names_tuple[i][1] for i in range(lim)]
        return items

    @staticmethod
    def map_entities_to_database(entities: Entities) -> Optional[str]:
        result = ""
        label_query = """MATCH (p)
        WHERE p.lemma = $value
        RETURN p.lemma AS result, labels(p)[0] AS type
        """

        for entity in entities.names:
            responses = graph.query(label_query, {"value": entity})
            for response in responses:
                result += f"{entity} knowledge-base-अन्तः भवति :{response['type']} {{'lemma': {response['result']}}} इति । अस्य सम्बन्धानि (relationships) अधः वर्तन्ते\n"
                mapped_entity = {'type': response['type'], 'lemma': response['result']}

                edge_outward = f"""MATCH (p:{mapped_entity['type']})-[r]->(q)
                WHERE p.lemma = '{mapped_entity['lemma']}'
                RETURN r AS relationship, labels(q)[0] AS type, q.lemma AS lemma
                """
            
                edge_inward =f"""MATCH (p:{mapped_entity['type']})<-[r]-(q)
                WHERE p.lemma = '{mapped_entity['lemma']}'
                RETURN r AS relationship, labels(q)[0] AS type, q.lemma AS lemma
                """

                def prune_edges(edges):
                    out = {}
                    for edge in edges:
                        rel = edge['relationship'][1]
                        if rel not in out:
                            out[rel] = []
                        edge_rep = f"{edge['type']}"# {{'lemma': {edge['lemma']}}}"
                        if edge_rep not in out[rel]:
                            out[rel].append(edge_rep)
                    return out
                
                outward_edges = graph.query(edge_outward)
                result += f"अस्मात् बहिः गच्छन्ति (outgoing) सम्बन्धानि -\n"
                if len(outward_edges) > MAX_WID_PER_ELEM:
                        outward_edges = random.sample(outward_edges, MAX_WID_PER_ELEM)
                for e, t in prune_edges(outward_edges).items():
                    result += f"-[:{e}]->({'|'.join([':'+l for l in t])})\n"
                
                inward_edges = graph.query(edge_inward)
                result += f"अस्मात् अन्तः गच्छन्ति (incoming) सम्बन्धानि -\n"
                if len(inward_edges) > MAX_WID_PER_ELEM:
                        inward_edges = random.sample(inward_edges, MAX_WID_PER_ELEM)

                for e, t in prune_edges(inward_edges).items():
                    result += f"<-[:{e}]-({'|'.join([':'+l for l in t])})\n"

                result += '\n'
    
        return result

    @staticmethod
    def map_relations_to_database(relations: Relationships, entities: Entities) -> Optional[str]:
        result_txt = ""
        track_paths = {}
        label_query = """MATCH (p)
        WHERE p.lemma = $value
        RETURN p.lemma AS result, labels(p)[0] AS type
        """

        for entity, path in zip(entities.names, entities.paths):
            responses = graph.query(label_query, {"value": entity})
        
            for response in responses:
                result_txt += f"(:{response['type']} {{'lemma': {response['result']}}}) इति अस्य सम्बन्धानि (relationships) अधः वर्तन्ते\n"
                mapped_entity = {'type': response['type'], 'lemma': response['result']}
                for relation in relations.names:
    
                    edge_outward = f"""MATCH (p:{mapped_entity['type']})-[:{relation}]->(q)
                    WHERE p.lemma = '{mapped_entity['lemma']}'
                    RETURN labels(q)[0] AS type, q.lemma AS lemma
                    """
                
                    edge_inward =f"""MATCH (p:{mapped_entity['type']})<-[:{relation}]-(q)
                    WHERE p.lemma = '{mapped_entity['lemma']}'
                    RETURN labels(q)[0] AS type, q.lemma AS lemma
                    """
                    
                    outward_edges = graph.query(edge_outward)
                    if len(outward_edges) > MAX_WID_PER_ELEM:
                        outward_edges = random.sample(outward_edges, MAX_WID_PER_ELEM)
                    
                    ends = [f":{edge['type']} {{'lemma': {edge['lemma']}}}" for edge in outward_edges]

                    for edge in outward_edges:
                        if edge['lemma'] not in track_paths:
                            track_paths[edge['lemma']] = []
                        track_paths[edge['lemma']].extend([f"{p}(:{response['type']} {{'lemma': {response['result']}}})-[:{relation}]->" for p in path])
            
                    if len(ends) > 0: result_txt += f"-[:{relation}]->({'|'.join(ends)})\n"
                    
                    inward_edges = graph.query(edge_inward)
                    if len(inward_edges) > MAX_WID_PER_ELEM:
                        inward_edges = random.sample(inward_edges, MAX_WID_PER_ELEM)
                    
                    ends = [f":{edge['type']} {{'lemma': {edge['lemma']}}}" for edge in inward_edges]

                    for edge in inward_edges:
                        if edge['lemma'] not in track_paths:
                            track_paths[edge['lemma']] = []
                        track_paths[edge['lemma']].extend([f"{p}(:{response['type']} {{'lemma': {response['result']}}})<-[:{relation}]-" for p in path])
                
                    if len(ends) > 0: result_txt += f"<-[:{relation}]-({'|'.join(ends)})\n"
    
                result_txt += '\n'
        
        return result_txt, track_paths


    def __init__(self, model= 'gpt-4o', dataset='sanskrit', lemmatizer=None, d=MAX_DEPTH):
        self.model = get_chat_model_kg(model)
        self.lemmatizer = lemmatizer
        self.d = d
        self.dataset = dataset
        self.topic_entity_chain = self.Topic_entity_chain()
        self.relation_chain = self.Relation_chain()
        self.entity_chain = self.Entity_chain()
        self.reasoning_chain = self.Reasoning_chain()
        self.answer_chain = self.Answer_chain()
    
    
    def __call__(self, input):

        inp = {'question': f"{input['QUESTION']} {input['CHOICES']}"}
        ## Extract entities from the text
        current_entities, flag = chain_invoke_(self.topic_entity_chain, inp)
        out = {'max_d': self.d}

        if flag:
            current_entities = None

        for i in range(self.d):

            if flag:
                out['max_d'] = i+1
                break

            inp['relations'] = self.map_entities_to_database(current_entities)
            
            ## Assign relavance scores to the relations
            current_relations, flag = chain_invoke_(self.relation_chain, inp)

            if flag:
                out['max_d'] = i+1
                break

            inp['relations'], track_paths = self.map_relations_to_database(current_relations, current_entities)

            ## Assign relavance scores to the new entities
            current_entities, flag = chain_invoke_(self.entity_chain, inp)
            
            if flag:
                out['max_d'] = i+1
                break


            ## append track path information
            current_entities.append_paths(track_paths)
            current_entities = self.prune(current_entities)

            inp['entities'] = current_entities
            decision, _ = chain_invoke_(self.reasoning_chain, inp)

            try:
                decision = int(decision)
            except:
                decision = 0
            
            if decision:
                out['max_d'] = i+1
                break

        inp['entities'] = current_entities
        out['answer'], flag = chain_invoke_(self.answer_chain, inp)
        if flag:
            out['answer'] = 'ERROR'
        out['paths'] = self.path_info_format(inp)['path_info'].replace("\n", "; ")

        return out



    def Topic_entity_chain(self):
        topic_entity_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """त्वम् knowledge-graph-तः उत्तराणि निष्कर्षयितुं प्रश्नात् entities विन्दसि च तानि सह relevance-score (0-1 मध्ये) समर्पयसि ।
                    output उदाहरणम् ('रामः', 0.8), ('सीता', 0.7) । ततो विवृतं मा कुरु । """,
                ),
                (
                    "human",
                    "प्रश्नः: {question}",
                ),
            ]
        )
        return topic_entity_prompt | self.model.with_structured_output(Entities) | RunnableLambda(self.lemmatizer.entity_lemmas) | RunnableLambda(self.prune)

    def Relation_chain(self):
        relation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """त्वम् दत्त-प्रश्नस्य उत्तराणि knowledge-graph-तः निष्कर्षितुं knowledge-graph-तः इदानीं पर्यन्तं निष्कर्षित-सम्बन्धेभ्यः अवश्यानि सम्बन्धानि सह relevance-score (0-1 मध्ये) समर्पयसि ।
                    output उदाहरणम् ('IS_FATHER_OF', 0.8), ('IS_CROSSED_BY', 0.7), ... । ततो विवृतं मा कुरु । """,
                ),
                (
                    "human",
                    """प्रश्नः: {question}
                    निष्कर्षितानि सम्बन्धानि: {relations}""",
                ),
            ]
        )
        return relation_prompt | self.model.with_structured_output(Relationships) | RunnableLambda(self.prune)
    
    def Entity_chain(self):
        entity_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """त्वम् दत्त-प्रश्नस्य उत्तराणि knowledge-graph-तः निष्कर्षितुं knowledge-graph-तः इदानीं पर्यन्तं निष्कर्षित-सम्बन्धेभ्यः अवश्यानि nodes (lemmas) सह relevance-score (0-1 मध्ये) समर्पयसि ।
                    output उदाहरणम् ('राम', 0.8), ('सीता', 0.7) । ततो विवृतं मा कुरु । """,
                ),
                (
                    "human",
                    """प्रश्नः: {question}
                    निष्कर्षितानि सम्बन्धानि: {relations}""",
                ),
            ]
        )


        return entity_prompt | self.model.with_structured_output(Entities) | RunnableLambda(lambda x: self.lemmatizer.entity_lemmas(x, lemmatized=True))
    
    @staticmethod
    def path_info_format(inp):
            path_info = []
            if inp['entities'] == None:
                inp['path_info'] = ''
                return inp
            for e, p in zip(inp['entities'].names, inp['entities'].paths):
                for p0 in p:
                    path_info.append(f"{p0}({{'lemma': {e}}})")

            inp['path_info'] = '\n'.join(path_info)
            return inp

    def Reasoning_chain(self):
        reasoning_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """त्वम् दत्त-प्रश्नस्य उत्तराणि knowledge-graph-तः निष्कर्षितुं knowledge-graph-तः इदानीं पर्यन्तं निष्कर्षितं यत्-किञ्चिद् प्रश्नस्य उत्तरं दातुं अलम् (1) वा नालम् (0) इति वक्तव्यम्।
                    output 1 अथवा 0 । न अन्यत् वदसि""",
                ),
                (
                    "human",
                    """प्रश्नः: {question}
                    निष्कर्षितम्: {path_info}""",
                ),
            ]
        )

        
        return RunnableLambda(self.path_info_format) | reasoning_prompt | self.model
    
    def Answer_chain(self):
        dataset = 'आयुर्वेद' if self.dataset == 'ayurveda' else 'रामायण'
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""अधः {dataset}-सम्बन्धे पृष्ट-प्रश्नस्य प्रत्युत्तरं देहि। तदपि प्रश्नोचितविभक्तौ भवेत् न तु प्रातिपदिक रूपे । तदपि एकेनैव पदेन यदि उत्तरे कारणं नापेक्षितम्। कथम् किमर्थम् इत्यादिषु एकेन लघु वाक्येन उत्तरं देहि अत्र तु एक-पद-नियमः नास्ति। 
                    अपि च यथाऽवश्यम् अधः दत्तैः knowledge-graph-तः निष्कर्षित-विषयैः सहाय्यं गृहाण। तत्तु सर्वदा साधु इति नाऽस्ति प्रतीतिः। उत्तरम् यावद् लघु शक्यं तावत् लघु भवेत्""",
                ),
                (
                    "human",
                    """प्रश्नः: {question}
                    निष्कर्षितम्: {path_info}
                    उत्तरम्: """,
                ),
            ]
        )

        return RunnableLambda(self.path_info_format) | answer_prompt | self.model | StrOutputParser() | RunnableLambda(lambda x: x.replace('।','').strip())

    
def run_kgqa(in_file, model, lemmatizer, d= 3, out_file=None, force=None):
    dataset = os.path.split(in_file)[-1].replace(".tsv","")
    if out_file==None:
        out_pth = "results/kgqa"
        os.makedirs(out_pth, exist_ok=True)
        out_file = os.path.join(out_pth, f"{dataset}.tsv")
    
    in_df = pd.read_csv(in_file, sep='\t')
    if os.path.exists(out_file):
        out_df = pd.read_csv(out_file, sep='\t')
    else:
        out_df = in_df.copy()

    tog = ToG(model=model, dataset=dataset, lemmatizer=lemmatizer) 
    if model in out_df.columns and (not force):
        print(f"Warning! column {model} already exists in {out_file}. Skipping the chain. Specify -f to overwrite.")
    else:
        print(f"Applying ToG for KGQA on {in_file} with {model}, saving to {out_file}")
        in_df[model] = in_df.progress_apply(tog, axis=1)
        out_df[model] = in_df.apply(lambda x: x[model]['answer'], axis=1)
        out_df[model+"_maxd"] = in_df.apply(lambda x: x[model]['max_d'], axis=1)
        out_df[model+"_paths"] = in_df.apply(lambda x: x[model]['paths'], axis=1)


    out_df.to_csv(out_file, sep='\t', index=False)

    return


def run_kgqa_default(in_file, model= None, depth=MAX_DEPTH, **kwargs):
    if model == None:
        models = KG_DEFAULT_MODELS
    else:
        models = [model]
    
    lemmatizer = Lemmatizer()
    for m in models:
        run_kgqa(in_file, m, lemmatizer=lemmatizer, d=depth, **kwargs)
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run KGQA with LLM Agent by Think-on-Graph (ToG)")
    parser.add_argument('-i','--in-file',type=str, default= 'data/qa_set/sanskrit.tsv',help="Path to input tsv file with columns QUESTION, ANSWER")
#    parser.add_argument('-l','--lang', type=str, help="Language of input file, by default deduces from [IN] assuming format [LANG].tsv")
    parser.add_argument('-m','--model',type=str, help= "LLM model name, currently supports: gpt-*, claude-*, gemini-*, mistral-*, llama-*, by default runs on gpt-4o, claude-3-5-sonnet, gemini-1.5-pro, mistral-large and llama-v3p1-8b-instruct")
    parser.add_argument('-o','--out-file',type=str, help="out file name to store predictions, by default stores in results/kgqa/[LANG].tsv")
    parser.add_argument('-f','--force', action='store_true', help="overwrite the outfile column")
    parser.add_argument('-d', '--depth', type=int, default=MAX_DEPTH, help="Graph traversal depth")
    args = parser.parse_args()

    args_dict = vars(args)
    run_kgqa_default(**args_dict)

