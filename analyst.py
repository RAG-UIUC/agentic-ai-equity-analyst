from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
import os

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

collection = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="company_filings", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

model = init_chat_model("gpt-4o", model_provider="openai")

@tool
def analyze(query):
    """Analyze the query using the data fetched in the database
    
    """
    res = collection.similarity_search(query=query, k=10)

    messages = [SystemMessage(content="You are a professional technical financial analyst."),
                HumanMessage(content=f"Summarize and analyze the following data: {res[:]} . Do not repeat yourself"),
                ]
    
    return model.invoke(messages).content

#print(analyze("Apple's revenue growth in 2024"))