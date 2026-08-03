import json
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---- 1. Load parsed subfolder data ----
INPUT_FILE = "mplus_health_topics_knowledge_base.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"[INFO] {len(raw_data)} entries loaded from {INPUT_FILE}.")

# ---- 2. Convert to LangChain Document objects ----
documents = [
    Document(
        page_content=f"Condition: {entry['condition']}\nQuestion: {entry['question']}\nAnswer: {entry['content']}",
        metadata={
            "condition": entry["condition"],
            "question": entry["question"],
            "source": entry.get("source", "MedlinePlus Health Topics")
        }
    )
    for entry in raw_data
]

# ---- 3. Load FastEmbed Model ----
print("[INFO] Loading FastEmbed model...")
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ---- 4. Build FAISS Index in Batches ----
print("[INFO] Building FAISS index for MedlinePlus Health Topics...")
BATCH_SIZE = 500
vector_store = None

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    if vector_store is None:
        vector_store = FAISS.from_documents(batch, embedding_model)
    else:
        vector_store.add_documents(batch)
    print(f"  -> Processed {min(i + BATCH_SIZE, len(documents))}/{len(documents)}")

# ---- 5. Save FAISS Index locally ----
INDEX_SAVE_PATH = "faiss_mplus_health_topics"
vector_store.save_local(INDEX_SAVE_PATH)
print(f"[SUCCESS] FAISS index saved to ./{INDEX_SAVE_PATH}/")

# ---- Quick Test ----
print("\n[TEST] Query: 'I have headache and sensitivity to light'")
results = vector_store.similarity_search("I have headache and sensitivity to light", k=2)
for r in results:
    print(f"  -> {r.metadata['condition']}: {r.page_content[:100]}...")