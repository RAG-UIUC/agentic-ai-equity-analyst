import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI


sonar_client = OpenAI(
    api_key="PERPLEXITY_API_KEY",
    base_url="https://api.perplexity.ai"
)

# Query Sonar for latest financial info
sonar_response = sonar_client.chat.completions.create(
    model="sonar-pro",
    messages=[{"role": "user", "content": "Latest Apple earnings press release"}]
)
financial_text = sonar_response.choices[0].message.content

openai.api_key = "OPENAI_API_KEY"
EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(texts):
    resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=texts)
    return [e["embedding"] for e in resp["data"]]

# Chroma setup
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection("financial_news")

# Embed and store
embedding = embed_text([financial_text])[0]
collection.add(
    embeddings=[embedding],
    documents=[financial_text],
    ids=["unique_id_for_this_entry"]  # Use a unique ID for each new entry
)

query = "Apple's revenue growth in the last quarter"
query_emb = embed_text([query])[0]

results = collection.query(
    query_embeddings=[query_emb],
    n_results=5,
    include=["documents", "distances", "ids"]
)

for i, (doc, dist, doc_id) in enumerate(zip(results['documents'][0], results['distances'][0], results['ids'][0])):
    similarity = 1 - dist
    print(f"Result {i+1}: {doc} (Similarity: {similarity:.4f}, ID: {doc_id})")


openai_client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
report = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": f"Summarize and analyze this data: {results['documents'][0][0]}"}
    ]
)
print(report.choices[0].message.content)

