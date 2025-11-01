import uuid, os
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from datetime import date, timedelta, datetime, timezone
from zoneinfo import ZoneInfo

company = "Apple"
symbol = "AAPL"

BATCH = 300

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
chroma_api_key = os.getenv("CHROMA_API_KEY")
chroma_tenant = os.getenv("CHROMA_TENANT")
chroma_database = os.getenv("CHROMA_DATABASE")

ticker = yf.Ticker(symbol)
data = ticker.history(start= date.today() - timedelta(days=1), end=date.today(),
    interval="1m",
    auto_adjust=True,
    prepost=True
)

if data.index.tz is None:
    data = data.tz_localize("UTC")

ny_tz = ZoneInfo("America/New_York")

texts = []
metadatas = []
ids = []

for ts_utc, row in data.iterrows():
    text = (
        f"timestamp_utc={ts_utc.isoformat()} "
        f"Open={row['Open']} High={row['High']} Low={row['Low']} Close={row['Close']} "
        f"Volume={row['Volume']} Dividends={row.get('Dividends', 0.0)} "
        f"StockSplits={row.get('Stock Splits', 0.0)}"
    )
    texts.append(text)

    ts_ny = ts_utc.tz_convert(ny_tz)
    meta = {
        "symbol": symbol,
        "company": company,
        "date_retrieved": date.today().isoformat(),
        "time_retrieved": datetime.now(timezone.utc).astimezone(ny_tz).strftime("%H:%M:%S"),
        "date_published": ts_ny.date().isoformat(), 
        "time_published": ts_ny.strftime("%H:%M:%S"),    
        "source": "yf",
        "url": "url",
    }
    metadatas.append(meta)
    ids.append(str(uuid.uuid4()))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

vector_store = Chroma(
    collection_name="financial_data",
    embedding_function=embeddings,
    chroma_cloud_api_key=chroma_api_key,
    tenant=chroma_tenant,
    database=chroma_database,
)

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

for t_chunk, m_chunk, id_chunk in zip(
        chunked(texts, BATCH),
        chunked(metadatas, BATCH),
        chunked(ids, BATCH),
    ):
    vector_store.add_texts(texts=t_chunk, metadatas=m_chunk, ids=id_chunk)

