from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import List
import json

load_dotenv()

# Setup local Hugging Face embeddings and Ollama LLM
persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3.2", temperature=0)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Pydantic model for structure validation
class QueryVariations(BaseModel):
    queries: List[str]

# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations (Robust JSON Method)
# ──────────────────────────────────────────────────────────────────

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents.

Original query: {original_query}

You MUST output your response strictly as a valid JSON object matching this exact format, with no extra text or markdown code blocks:
{{"queries": ["variation 1", "variation 2", "variation 3"]}}"""

messages = [
    SystemMessage(content="You are a helpful assistant that outputs only valid JSON."),
    HumanMessage(content=prompt)
]

try:
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Clean up potential markdown formatting blocks from the local model
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    # Parse JSON safely into our Pydantic structure
    parsed_data = json.loads(content)
    query_variations = parsed_data.get("queries", [original_query])

except Exception as e:
    print(f"⚠️ JSON parsing failed ({e}), falling back to original query.")
    query_variations = [original_query]

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "="*60)

# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")