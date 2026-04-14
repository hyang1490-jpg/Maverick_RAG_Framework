"""
Debug Search - ChromaDB 检索结果诊断脚本
"""

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "icarus_failures_final"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL_NAME = "bge-m3"

TEST_QUERY = "我要砸500万做一个社交平台挑战微信"


def main() -> None:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embedding_fn = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL_NAME,
        url=OLLAMA_EMBED_URL,
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    print(f"Collection: {COLLECTION_NAME} | Total: {collection.count()}")
    print(f"Query: {TEST_QUERY}")
    print("=" * 70)

    results = collection.query(
        query_texts=[TEST_QUERY],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        print(f"\n--- Result {i} ---")
        print(f"  Distance : {dist}")
        print(f"  Metadata : {meta}")
        print(f"  Document : {doc}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
