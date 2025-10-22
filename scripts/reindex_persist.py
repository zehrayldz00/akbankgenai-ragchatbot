import os
from dotenv import load_dotenv
from typing import List
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download
from sentence_transformers import SentenceTransformer
import chromadb
try:
    from chromadb.config import Settings
except Exception:
    Settings = None

load_dotenv()

HF_DATASET       = os.getenv("HUGGINGFACE_DATASET", "zehrayldz00/istanbulguide")
HF_TEXT_FILE     = os.getenv("HUGGINGFACE_TEXT_FILE", "")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "istanbulguide_collection")
PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_persist")

# 🔁 Aynı multilingual embedding:
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "64"))

def create_chroma_client(persist_path: str):
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=persist_path)
    if Settings is None:
        raise SystemExit("Chroma version unsupported.")
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_path)
    return chromadb.Client(settings=settings)

def choose_txt_in_hf_repo(repo_id: str, preferred: str = "") -> str:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    txts = [p for p in files if p.lower().endswith(".txt")]
    if not txts:
        raise SystemExit(f"No .txt file in dataset '{repo_id}'.")
    if preferred:
        if preferred in files:
            return preferred
        cand = [p for p in txts if p.endswith(preferred)]
        if cand:
            return cand[0]
        print(f"Warning: '{preferred}' not found. Using first .txt in repo.")
    return txts[0]

def load_text_from_hf(repo_id: str, file_in_repo: str) -> str:
    fp = hf_hub_download(repo_id=repo_id, filename=file_in_repo, repo_type="dataset")
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    chunks, i, L = [], 0, len(text)
    while i < L:
        chunk = text[i : i + size].strip()
        if chunk:
            chunks.append(chunk)
        i += size - overlap if size > overlap else size
    return chunks

def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    print(">> Embedding model:", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print(">> Chroma opening:", PERSIST_DIR)
    client = create_chroma_client(PERSIST_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f">> Old collection deleted: {COLLECTION_NAME}")
    except Exception:
        pass

    print(f">> Creating collection: {COLLECTION_NAME}")
    if hasattr(client, "create_collection"):
        collection = client.create_collection(COLLECTION_NAME)
    else:
        collection = client.get_or_create_collection(COLLECTION_NAME)

    print(f">> Reading HF dataset: {HF_DATASET}")
    txt_name = choose_txt_in_hf_repo(HF_DATASET, HF_TEXT_FILE)
    print(f">> Using file: {txt_name}")
    raw_text = load_text_from_hf(HF_DATASET, txt_name)

    print(">> Chunking text...")
    chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise SystemExit("No chunks produced (empty text?).")

    print(f">> Total chunks: {len(chunks)}")
    all_ids   = [f"{COLLECTION_NAME}_{i:07d}" for i in range(len(chunks))]
    all_metas = [{"source_row": i, "chunk_id": f"{i:07d}", "source_file": txt_name} for i in range(len(chunks))]

    print(">> Embedding + add to Chroma...")
    idx = 0
    for batch_ids, batch_docs, batch_metas in tqdm(
        zip(batched(all_ids, BATCH_SIZE), batched(chunks, BATCH_SIZE), batched(all_metas, BATCH_SIZE)),
        total=(len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    ):
        embs = embed_model.encode(list(batch_docs), convert_to_numpy=True).tolist()
        collection.add(ids=list(batch_ids), documents=list(batch_docs), metadatas=list(batch_metas), embeddings=embs)
        idx += len(batch_ids)

    print(f">> Done. Added chunks: {idx}")
    print(">> Now run: python chatbot.py")

if __name__ == "__main__":
    main()
