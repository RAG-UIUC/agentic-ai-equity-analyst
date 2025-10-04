import chromadb
import yfinance as yf
import pandas as pd
from datetime import datetime, date
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
  
client = chromadb.CloudClient(
  api_key='ck-yu7vxc2gHZuML9UYAzzHmvvWbEhgxvhxoskugYWi5kR',
  tenant='39d705f2-76cc-419f-adea-b71614d9aeb4',
  database='AIEquityAnalyst '
)
EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(texts):
    resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [e.embedding for e in resp.data]

ticker = yf.Ticker('AAPL')
data = ticker.history("1y")

docs = [str(row) for _, row in data.iterrows()]
emb_text = embed_text(docs)
ids = [str(i) for i in range(len(emb_text))]
col = client.get_collection("financial_data")
col.add(documents = docs, embeddings=emb_text, metadatas=[{"source":"yfinance"}]*len(docs), ids = ids)