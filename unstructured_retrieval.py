import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
chroma_api_key = os.getenv("CHROMA_API_KEY")
chroma_tenant = os.getenv("CHROMA_TENANT")
chroma_database = os.getenv("CHROMA_DATABASE")

ticker = yf.Ticker('AAPL')
data = ticker.history("1y")
docs = [str(row) for _, row in data.iterrows()]

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small", api_key=openai_api_key)

vector_store = Chroma(collection_name = "financial_data", embedding_function=embeddings, chroma_cloud_api_key=chroma_api_key, tenant=chroma_tenant, database=chroma_database)

EMBEDDING_MODEL = "text-embedding-3-small"

ids = [str(i) for i in range(len(docs))]
metadatas = [{"source": "yfinance"}] * len(docs)
vector_store.add_texts(texts=docs, metadatas=metadatas, ids=ids)