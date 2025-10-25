import requests
import uuid
from openai import OpenAI
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

# these lines read the filing and such 
# OH MY GOD CHUNKING !!!!!!!!!!!!!!! (REAL) 
from langchain_text_splitters import RecursiveJsonSplitter, RecursiveCharacterTextSplitter

filename = ""
doctype = ""

if per == "FY" or per == "Q4":
  filename = f"{ticker}_{year}_10-K_filing"
  doctype = "10-K filing"
else:
  filename = f"{ticker}_{year}_{per}_10-Q_filing"
  doctype = "10-Q filing"

url = f"https://financialmodelingprep.com/stable/financial-reports-json?symbol={ticker}&year={year}&period={per}&apikey={fmp_key}"
json_data = requests.get(url).json()
json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
txt_splitter = RecursiveCharacterTextSplitter()

chunks = json_splitter.split_json(json_data=json_data, convert_lists=True)
txt = chunks[0] 

unique_id = str(uuid.uuid4())

def flatten_json_to_text(obj):
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(f"{k}: {flatten_json_to_text(v)}")
        return ", ".join(parts)
    elif isinstance(obj, list):
        return ", ".join(flatten_json_to_text(x) for x in obj)
    
    return str(obj)

for txt in json_data:
  #print(txt)

  collection.add_texts(texts=[flatten_json_to_text(txt)], 
                       ids=[unique_id], 
                       metadatas=[ticker, year, per, doctype])

# testing stuffs
'''
query = "Apple's revenue in 2022"
res = collection.similarity_search(query=query, k=10)

# langchain conversion (this is here just for testing)
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("gpt-4o", model_provider="openai")

messages = [SystemMessage(content="You are a professional technical financial analyst."),
            HumanMessage(content=f"Summarize and analyze the following data:{res[:]}"),
            ]

print(model.invoke(messages).content)
'''