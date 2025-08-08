from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# initialize Pinecone only once
pc = Pinecone(api_key="pcsk_6gcBSp_2ufJs9GCNpZwN9pztv1ggbFNbJhr3w2E2wKSfyyB7iGqcuzee79apu8weRx9D7s")
index_name = "hackrx-rag"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)

def embed_and_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    metadatas = [{"doc_id": i, "text": chunk} for i, chunk in enumerate(chunks)]
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    to_upsert = [(f"id-{i}", embeddings[i], metadatas[i]) for i in range(len(chunks))]
    index.upsert(vectors=to_upsert)
    return chunks, metadatas
