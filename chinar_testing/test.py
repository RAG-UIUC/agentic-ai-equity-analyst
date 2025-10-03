import chromadb
import yfinance as yf
import pandas as pd
from datetime import datetime, date
  
client = chromadb.CloudClient(
  api_key='ck-yu7vxc2gHZuML9UYAzzHmvvWbEhgxvhxoskugYWi5kR',
  tenant='39d705f2-76cc-419f-adea-b71614d9aeb4',
  database='AIEquityAnalyst '
)

ticker = yf.Ticker('AAPL')
data = ticker.history("1y")

docs = [str(row) for _, row in data.iterrows()]
# col = client.create_collection("new_col")
# col.add(documents = docs, metadatas=[{"source":"yfinance"}]*len(docs), ids = ids)
ids = [str(i) for i in range(len(docs))]
col = client.get_collection("test_data")

