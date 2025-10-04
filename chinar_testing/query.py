import chromadb
from chromadb.config import Settings
import openai
import uuid
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
sonar_client = OpenAI(
    api_key= os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

# Query Sonar for latest financial info
sonar_response = sonar_client.chat.completions.create(
    model= "sonar",
    messages=[{"role": "user", "content": "Latest Apple earnings press release"}]
)
financial_text = sonar_response.choices[0].message.content

EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(texts):
    resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [e.embedding for e in resp.data]

# Chroma setup
client = chromadb.CloudClient(
  api_key='ck-yu7vxc2gHZuML9UYAzzHmvvWbEhgxvhxoskugYWi5kR',
  tenant='39d705f2-76cc-419f-adea-b71614d9aeb4',
  database='AIEquityAnalyst ')
collection = client.get_or_create_collection("test_data")

# Embed and store
embedding = embed_text([financial_text])[0]
unique_id = str(uuid.uuid4())
collection.add(
    embeddings=[embedding],
    documents=[financial_text],
    ids=[unique_id]
)

query = "Apple's revenue growth in the last quarter"
query_emb = embed_text([query])[0]

results = collection.query(
    query_embeddings=[query_emb],
    n_results=5,
    include=["documents", "distances"]
)

for i, (doc, dist, ) in enumerate(zip(results['documents'][0], results['distances'][0])):
    similarity = 1 - dist
    print(f"Result {i+1}: {doc} (Similarity: {similarity:.4f})")


openai_client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
report = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": f"Summarize and analyze this data: {results['documents'][0][0]}"}
    ]
)
print(report.choices[0].message.content)

