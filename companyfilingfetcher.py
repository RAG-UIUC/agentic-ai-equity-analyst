import requests
import pandas as pd
import uuid
from openai import OpenAI
from urllib.request import urlopen
import certifi, json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def embed_text(text):
    return embeddings.embed_documents(text)

'''chroma_client = chromadb.CloudClient(
  api_key=os.getenv("CHROMADB_API_KEY"),
  tenant=os.getenv("CHROMADB_TENANT"),
  database=os.getenv("CHROMADB"))'''

collection = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="company_filings",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

fmp_key = os.getenv("FMP_API_KEY")
year = 2022
ticker = "AAPL"
per = "FY" # Q4, FY are 10-K reports; Q1-3 are 10-Q reports

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

# these lines read the filing 
'''
url = f"https://financialmodelingprep.com/stable/financial-reports-xlsx?symbol={ticker}&year={year}&period={per}&apikey={fmp_key}"

xcel = requests.get(url) # 1 api call to fmp

read = pd.read_excel(xcel.content, sheet_name=None)
filename = ""

if per == "FY" or per == "Q4":
  filename = f"{ticker}_{year}_10-K_filing"
else:
  filename = f"{ticker}_{year}_{per}_10-Q_filing"

for k, v in read.items():
  v.to_csv(k, index=None, header=True)
  data = pd.read_csv(k)
  df = pd.DataFrame(data)

  for row in data.iterrows():
    txt = row[1].to_string()
    unique_id = str(uuid.uuid4())
    collection.add(embeddings=[embed_text(txt)[0]], documents=[f"{filename} : {txt}"], ids=[unique_id])
'''

query = "Apple's revenue in 2022"
res = vector_store.similarity_search(query=query, k=10)


# langchain conversion (this is here just for testing)
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("gpt-4o", model_provider="openai")

messages = [SystemMessage(content="You are a professional technical financial analyst."),
            HumanMessage(content=f"Summarize and analyze the following data:{res[:]}"),
            ]

print(model.invoke(messages).content)
