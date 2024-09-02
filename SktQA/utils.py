from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_mistralai import ChatMistralAI
from langchain_fireworks import ChatFireworks
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import os

tqdm.pandas()
load_dotenv()

MAX_LENGTH = 64
MAX_K = 4
DEFAULT_MODELS = ['gpt-4o', 'mistral-large-latest', 'llama-v3p1-405b-instruct', 'gemini-1.5-pro', 'claude-3-5-sonnet-20240620']
LOW_END_MODELS = ['gpt-4o-mini', 'gpt-3.5-turbo', 'gemini-1.0-pro', 'llama-v3p1-70b-instruct', 'llama-v3p1-8b-instruct']

def get_chat_model(model):
    if model in ['gpt-4o','gpt-4o-mini','gpt-3.5-turbo']:
        chat_model = ChatOpenAI(model=model, max_tokens= MAX_LENGTH)
    elif model in ['claude-3-5-sonnet-20240620']:
        chat_model = ChatAnthropic(model=model, max_tokens= MAX_LENGTH)
    elif model in ['gemini-pro', 'gemini-1.5-pro']:
        chat_model = ChatVertexAI(model=model, max_tokens= MAX_LENGTH)
    elif model in ['mistral-large-latest']:
        chat_model = ChatMistralAI(model=model,api_key=os.environ['MISTRAL_API_KEY'], max_tokens= MAX_LENGTH)
    elif model in ['llama-v3p1-405b-instruct', 'llama-v3p1-70b-instruct','llama-v3p1-8b-instruct']:
        chat_model = ChatFireworks(model=f"accounts/fireworks/models/{model}", api_key=os.environ['FIREWORKS_API_KEY'], max_tokens= MAX_LENGTH)
    else:
        print(f"Error: Unsupported model - {model}")
        exit(1)

    return chat_model 

def get_chat_model_rag(model):
    if model in ['gpt-4o','gpt-4o-mini','gpt-3.5-turbo']:
        chat_model = ChatOpenAI(model_name=model, temperature=0, max_tokens= MAX_LENGTH)
    elif model in ['claude-3-5-sonnet-20240620']:
        chat_model = ChatAnthropic(model_name=model, temperature=0, max_tokens= MAX_LENGTH)
    elif model in ['gemini-pro', 'gemini-1.5-pro']:
        chat_model = ChatVertexAI(model_name=model, temperature=0, max_tokens= MAX_LENGTH)
    elif model in ['mistral-large-latest']:
        chat_model = ChatMistralAI(model_name=model,api_key=os.environ['MISTRAL_API_KEY'], temperature=0, max_tokens= MAX_LENGTH)
    elif model in ['llama-v3p1-405b-instruct', 'llama-v3p1-70b-instruct','llama-v3p1-8b-instruct']:
        chat_model = ChatFireworks(model_name=f"accounts/fireworks/models/{model}", api_key=os.environ['FIREWORKS_API_KEY'], temperature=0, max_tokens= MAX_LENGTH)
    else:
        print(f"Error: Unsupported model - {model}")
        exit(1)

    return chat_model

def chain_invoke(chain, inp):
    done = False
    ret = None
    i = 0
    while not done:
        try:
            done = True
            ret =  chain.invoke({"question": inp['QUESTION'], "choices": inp['CHOICES']})
        except Exception as e:
            done = False
            print(e)
            i += 1
            print(f"Reattempting {i}th time in a row...")

    return ret

def ApplyQAChainOnDF(df, chain, col_name='predicted'):
    df[col_name] = df.progress_apply(lambda x: chain_invoke(chain,x), axis=1)
    return df

diatrics_corr = {'r'+'̣':'ṛ', 's'+'̣':'ṣ', 'r'+'̣'+'̄': 'ṝ', 't'+'̣':'ṭ', 'd'+'̣':'ḍ', 
                 'n'+'̣':'ṇ', 'l'+'̱':'ḻ', 'a'+'̄':'ā', 'i'+'̄':'ī', 'u'+'̄':'ū', 's'+'́':'ś',
                 'n'+'̇': 'ṅ', 'n'+'̃' : 'ñ',
                }

def corr_diatrics(sent):
    new_sent = ''
    sent = list(sent)
    i = 0
    while i<len(sent):
        if i+1 < len(sent):
            c2 = sent[i] + sent[i+1]
            if c2 in diatrics_corr:
                new_sent += diatrics_corr[c2]
                i += 2
                continue
        new_sent += sent[i]
        i += 1
    return new_sent
