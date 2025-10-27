import requests, uuid, os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

'''INPUT FACTORS'''
company = "Apple"
year = "2024"

EMBEDDING_MODEL = "text-embedding-3-small"
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_api_key)
parser_data = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="parser_data",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)
company_filings = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="company_filings",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)
yfinance_data = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="financial_data",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small", api_key=openai_api_key)
model = init_chat_model("gpt-4o", model_provider="openai")
txt_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=0.35)
MIN_CHUNK_SZ = 3 
CHUNK_NO = 5 
TXT_THRESHOLD = 200

def chunk_text(txt):
  if (len(txt) > TXT_THRESHOLD):
    return txt_splitter.split_text(txt)

  return [txt]

with open("parser_queries.txt", "r") as f:
    for line in f:
      spl = line.rstrip('\n').split(":")
      query = company + " " + spl[0] + " " + "in " + year
      print(query)
      if spl[1] == "yf":
        res = yfinance_data.similarity_search(query=query, k=10)
      else:
        res = company_filings.similarity_search(query=query, k=10)
      messages = [SystemMessage(content="You are a professional financial equity analyst."),
                  HumanMessage(content=f"Summarize and analyze the following data: {res[:]} . Do not repeat yourself"),
                  ]
      
      txt = chunk_text(model.invoke(messages).content)
      print(txt)
      ids = [company + " " + year + " " + str(i) for i in range(len(txt))]
      parser_data.add_texts(texts=txt,metadatas=[{"company": company,"year": year} for _ in txt],ids=ids)


