from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
from indic_transliteration.sanscript import IAST, DEVANAGARI, transliterate
import torch
import importlib
from typing import Optional, cast, List
import numpy as np
import numpy.typing as npt
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import fasttext as ft
from gensim.models import KeyedVectors as kv
import regex as re
import argparse
from utils import *
import pickle
import string


## Custom embedding 'fasttext' or 'glove'
class VectorEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            model_name: str = "glove",
            cache_dir: Optional[str] = None,
    ):
        self._model_name = model_name
        if model_name == "glove":
            self._model = kv.load_word2vec_format("sa_embedding/models/glove/vectors.vec", binary=False)
            self._word_vec = lambda x: self._model[x]
            self._vector_size = self._model.vector_size
        elif model_name == "fasttext":
            self._model = ft.load_model("sa_embedding/models/fasttext/vectors.bin")
            self._word_vec = lambda x: self._model.get_word_vector(x)
            self._vector_size = self._model.get_dimension()
    
    
    @staticmethod
    def _normalize(vector: npt.NDArray) -> npt.NDArray:
        """Normalizes a vector to unit length using L2 norm."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
        
    def sentence2vec(self, sentence: str) -> npt.NDArray:
        words = sentence.split()
        word_vectors = []
        
        for word in words:
            if self._model_name == 'ft' or word in self._model:
                word_vectors.append(self._word_vec(word))
        
        if not word_vectors:
            return np.zeros(self._vector_size)
        
        # Compute the average vector
        avg_vector = np.mean(word_vectors, axis=0)
        
        return avg_vector


    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            embeddings.append(self.sentence2vec(text))
        return [e.tolist() for e in self._normalize(embeddings)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        return self([text])[0]


class Retriever:

    ## Loading lemmatizer
    checkpoint = 'mahesh27/t5lemmatizer'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    lemmatizer_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    stop_words = open("sa_embedding/stop_words.txt",'r').read().split()

    def __init__(self, file_path='data/ref/rAmAyaNa_dev.txt', load_cached=True):
        f_name = os.path.split(file_path)[1]
        if os.path.exists(f"models/{f_name}.pkl"):
            print("Found lemmatized docs in models/, skipping lemmatization")
            with open(f"models/{f_name}.pkl",'rb') as fp:
                self.splits = pickle.load(fp)
        else:
            # Load Documents
            loader = TextLoader(file_path=file_path)
            docs = loader.load()

            # Split
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            self.splits = text_splitter.split_documents(docs)
            print("Lemmatizing document chunks")
            for split in tqdm(self.splits):
                split.metadata['text'] = split.page_content.replace('\n\n','\n')
                split.page_content = self.lemmatize(split.page_content, translate_only=False)
            with open(f"models/{f_name}.pkl",'wb') as fp:
                pickle.dump(self.splits, fp)
        


    def lemmatize(self, chunk, translate_only=False):
        text = chunk.replace('\n\n','\n').split('\n')
        pattern = re.compile(r'[0-9]+')
        alphanumeric_pattern = re.compile(r'[a-zA-Z0-9]+')
        table = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation}
        lines = [alphanumeric_pattern.sub('',line) for line in text]
        lines = [pattern.sub('',transliterate(line, DEVANAGARI, IAST)) for line in lines]
        lines = [line.translate(table) for line in lines]
        if translate_only:
            return ' '.join(lines)
        tokenized_text = [{'input_ids': self.tokenizer(line)['input_ids'] + [self.tokenizer.eos_token_id]} for line in lines]
    
        inputs = self.data_collator(tokenized_text)
        with torch.no_grad():
            outputs = self.lemmatizer_model.generate(inputs=inputs['input_ids'], max_length=64)
    
        out_txt = corr_diatrics(' '.join(self.tokenizer.batch_decode(outputs, skip_special_tokens=True)))
        out_txt = out_txt.split()
        pre_processed = []
        for w in out_txt:
            if w not in self.stop_words:
                pre_processed.append(w)
        
        return ' '.join(pre_processed)

    def load_retriever(self, emb='bm25'):
            
        if emb == 'bm25':
            retriever = BM25Retriever.from_documents(self.splits)
        elif emb in ['fasttext', 'glove']:
            embedding = VectorEmbeddingFunction(model_name=emb)
            vectorstore = Chroma.from_documents(documents=self.splits, embedding=embedding)
            retriever = vectorstore.as_retriever()
        else:
            print("Error: invalid emb value, supported: 'fasttext', 'glove', 'bm25'")
            exit(1)
    
        self.retrieve = retriever

def RAGChain(model, retriever, dataset='sanskrit', k=2, context_only=False):
    text = 'आयुर्वेद' if dataset == 'ayurveda' else 'रामायण'
    template = f"""त्वया संस्कृत-भाषायाम् एव वक्तव्यम्। न तु अन्यासु भाषासु। अधः {text}-सम्बन्धे पृष्ट-प्रश्नस्य प्रत्युत्तरं देहि। तदपि एकेनैव पदेन, यावद् लघु शक्यं तावद्, तं पुनः विवृतम् मा कुरु। अपि च यथाऽवश्यम् अधः दत्त-सन्दर्भेभ्यः एकतमात् सहाय्यं गृहाण। तत्तु सर्वदा साधु इति नाऽस्ति प्रतीतिः।
     सन्दर्भाः:{{context}}
     प्रश्नः:{{question}} {{choices}}
    """
    prompt = PromptTemplate.from_template(template)
    # LLM
    llm = get_chat_model_rag(model=model)
    
    # Post-processing
    def format_docs(docs):
        return '\n\n'.join([doc.metadata['text'] for doc in docs[:k]])
    
    context_chain = RunnableLambda(lambda x: retriever.lemmatize(x['question'], translate_only=False)) | retriever.retrieve | format_docs
    llm_chain = prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.replace('।','').strip())


    if not context_only:
        rag_chain = (
            {"context": context_chain, "question": RunnablePassthrough(), "choices": RunnablePassthrough()}
            | RunnablePassthrough.assign(answer=llm_chain)
        )
    else:
        rag_chain = (
            RunnablePassthrough.assign(context=context_chain)
        )

    return rag_chain


def run_rag_qa(in_file, model, retriever, emb='bm25', k=2, out_file=None, force=None, context_only=False):
    dataset = os.path.split(in_file)[-1].replace(".tsv","")
    if out_file==None:
        out_pth = "results/rag"
        os.makedirs(out_pth, exist_ok=True)
        if dataset == 'ayurveda':
            pre = 'ayurveda_'
        else:
            pre = ''
        out_file = os.path.join(out_pth, f"{pre}{emb}_{k}.tsv")
    

    in_df = pd.read_csv(in_file, sep='\t')
    if os.path.exists(out_file):
        out_df = pd.read_csv(out_file, sep='\t', dtype=str)
    else:
        out_df = in_df.copy()
    
    out_df['ANSWER'] = in_df['ANSWER']

    chain = RAGChain(model=model, retriever=retriever, dataset=dataset, k=k, context_only=context_only)
    if model in out_df.columns and (not force):
        print(f"Warning! column {model} already exists in {out_file}. Skipping the chain. Specify -f to overwrite.")
    else:
        print(f"Applying RAG chain on {in_file} with {model}, saving to {out_file}")
        in_df = ApplyQAChainOnDF(in_df, chain=chain, col_name= model)
        for rank in range(k):
            if (f"context_{rank}" not in out_df.columns) or (force):
                out_df[f"new_context_{rank}"] = in_df.apply(lambda x: x[model]['context'].split('\n\n')[rank].replace('\t', ' ').replace('\n',' '), axis=1)
                if (f"context_{rank}" in out_df.columns and f"rel_{rank}" in out_df.columns):
                    out_df[f"rel_{rank}"] = out_df.apply(
                        lambda x:  [str(int(x[f"rel_{y}"])) for y in range(k) if x[f"context_{y}"] == x[f"new_context_{rank}"]] ,
                        axis=1)
                    out_df[f"rel_{rank}"] = out_df.apply(lambda x: x[f"rel_{rank}"][0] if len(x[f"rel_{rank}"])>0 else '', axis=1)

                out_df[f"context_{rank}"] = out_df[f"new_context_{rank}"].copy()
                del out_df[f"new_context_{rank}"]
                    
        if not context_only:
            out_df[model] = in_df.apply(lambda x: x[model]['answer'], axis=1)


    out_df.to_csv(out_file, sep='\t', index=False)
    return

def run_rag_default(in_file=None, model=None, emb=None, k=None, dataset=None, **kwargs):

    if in_file == None:
        in_file = 'data/qa_set/sanskrit.tsv'
    
    if model == None:
        models = DEFAULT_MODELS + LOW_END_MODELS
        models.remove('llama-v3p1-70b-instruct')
    else:
        models = [model]
    
    if emb == None:
        embs = ['bm25', 'fasttext', 'glove']
    else:
        embs = [emb]
    
    if k == None:
        ks = range(1,MAX_K+1)
    else:
        ks = [k]
    
    if dataset==None or dataset=='ramayana':
        retriever = Retriever()
    elif dataset=='ayurveda':
        in_file = 'data/qa_set/ayurveda.tsv'
        retriever = Retriever(file_path='data/ref/ayurveda_dev.txt')

    for e in embs:
        retriever.load_retriever(emb= e)
        for m in models:
            for ki in ks:
                run_rag_qa(in_file,m,retriever, emb=e,k=ki,**kwargs)

    return

 
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run RAG QA evaluation using LLMs")
    parser.add_argument('-i','--in-file', type=str, help="input file containing columns 'QUESTION', 'ANSWER' (gold). By default runs on all languages")
    parser.add_argument('-d','--dataset', type=str, help="Dataset either [ramayana] or [ayurveda]")
    parser.add_argument('-e','--emb',type=str, help="Embedding/retrieval method. Supported: fasttext, glove, bm25.")
    parser.add_argument('-k', '--k', type=int, help="Retriever number of documents threshold")
    parser.add_argument('-m','--model',type=str, help= "LLM model name, currently supports: gpt-*, claude-*, gemini-*, mistral-*, llama-*, by default runs on gpt-4o, claude-3-5-sonnet, gemini-1.5-pro, mistral-large and llama-v3p1-8b-instruct")
    parser.add_argument('-o','--out-file',type=str, help="out file name to store predictions, by default stores in results/rag/[LANG].tsv")
    parser.add_argument('-f','--force', action='store_true', help="overwrite the outfile column")
    args = parser.parse_args()

    args_dict = vars(args)
    run_rag_default(**args_dict)
