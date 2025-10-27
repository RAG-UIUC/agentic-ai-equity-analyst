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

EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

collection = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="testing", # <-- PAY ATTENTION HERE
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

fmp_key = os.getenv("FMP_API_KEY")
year = 2022
ticker = "AAPL"
per = "FY" # Q4, FY are 10-K reports; Q1-3 are 10-Q reports

model = init_chat_model("gpt-4o", model_provider="openai")

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
json_splitter = RecursiveJsonSplitter(max_chunk_size=1000)
txt_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=0.35)
MIN_CHUNK_SZ = 3 
CHUNK_NO = 5 
TXT_THRESHOLD = 200 # This is a good value. Please do not touch it. It's mine. - William 

def chunk_text(txt):
  if (len(txt) > TXT_THRESHOLD):
    return txt_splitter.split_text(txt)

  return [txt]

def summary(text):
  # i don't even know if the llm is going to follow these directions properly LOL 
  message = [SystemMessage(content="You are a helpful, punctual summarizer."),
              HumanMessage(content=f"Succinctly summarize the following: {text} (within 300 characters or less) while retaining all relevant details"),
              ]

  return model.invoke(message).content

def parse_json(obj, par_str, parent_id):
  chunks = [] 

  if isinstance(obj, dict): 
    items = list(obj.items())

    if len(items) >= CHUNK_NO * MIN_CHUNK_SZ:
      for i in range(CHUNK_NO):
        par = f"{par_str} -> "
        cur_id = str(uuid.uuid4())
        alt = ""

        for j in range(CHUNK_NO*i, min(len(items), CHUNK_NO*(i+1))):
          k = items[j][0]
          v = items[j][1] 
          par += f"{k}, "

          if isinstance(v, (dict, list)): 
            chunks.extend(parse_json(v, f"{par_str} -> {k}", cur_id))
          elif isinstance(v, (int, float, bool)):
            alt += f"{k} : {v}, "
          elif isinstance(v, str):
            if str(v).isspace() == False:        
              for i in chunk_text(str(v)):
                chunks.append((f"{par_str} -> {k} : {i}", str(uuid.uuid4()), cur_id))

        if len(alt) > 0:
          chunks.append((f"{par_str} -> {alt}", str(uuid.uuid4()), cur_id))

        chunks.append((par, cur_id, parent_id))
    else:
      par = f"{par_str} -> "
      cur_id = str(uuid.uuid4())
      alt = ""

      for k, v in obj.items(): # i have to admit this code could be written better but i'll go fix it later probably
        par += f"{k}, "

        if isinstance(v, (dict, list)): 
          chunks.extend(parse_json(v, f"{par_str} -> {k}", cur_id))
        elif isinstance(v, (int, float, bool)):
          alt += f"{k} : {v}, "
        elif isinstance(v, str):
          if str(v).isspace() == False:        
            for i in chunk_text(str(v)):
              chunks.append((f"{par_str} -> {k} : {i}", str(uuid.uuid4()), cur_id))

      if len(alt) > 0:
        for i in chunk_text(alt):
          chunks.append((f"{par_str} -> {i}", str(uuid.uuid4()), cur_id))

      chunks.append((par, cur_id, parent_id))

  elif isinstance(obj, list):  
    alt = ""
    
    for i in range(len(obj)): 
      if isinstance(obj[i], (int, float, bool, str)):
         alt += f"{i} : {obj[i]}, "
      else:
        chunks.extend(parse_json(obj[i], par_str, parent_id))

    if len(alt) > 0: 
      for i in chunk_text(alt):
        chunks.append((f"{par_str} -> {i}", str(uuid.uuid4()), parent_id))

  else: 
    if str(obj).isspace() == False:
      for i in chunk_text(str(obj)):
        chunks.append((par_str + i, str(uuid.uuid4()), parent_id))
    
  return chunks 


def embed_json(json_data):
  for i, cur_id, par_id in parse_json(json_data, "", 0):
    collection.add_texts(texts=[i], 
                        ids=[cur_id],
                        metadatas=[{"parent" : par_id, "ticker" : ticker, "year" : year, "period" : per, "type" : doctype}])

# embed_json(json_data)
