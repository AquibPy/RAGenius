import argparse
import asyncio
from dotenv import load_dotenv
from api.src.data_loader import load_all_documents
from api.src.vectorstore import ChromaVectorStore
from api.src.search import RAGEngine  # unified RAG class

load_dotenv(override=True)


def build_vectorstore():
    print("[STEP] Initializing Chroma Vector Store...")
    store = ChromaVectorStore(persist_directory="chromadb_store")

    if store.collection.count() == 0:
        print("[STEP] Loading documents from 'data' directory...")
        docs = load_all_documents("data")
        store.add_documents(docs)
        print("[DONE] Documents added successfully.")
    else:
        print(f"[INFO] Collection already has {store.collection.count()} documents.")

    return store


def run_basic_rag(query: str):
    """Run the basic (non-streaming) RAG pipeline."""
    print(f"[STEP] Running Basic RAG Search for query: '{query}'")
    rag = RAGEngine()
    result = rag.query(query, top_k=3, summarize=True)
    
    print("\n🧠 Answer:\n", result.get("answer", "No answer generated."))
    print("\n📜 History:\n", result.get("history", []))


async def run_streaming_rag(query: str):
    """Run the streaming RAG pipeline asynchronously."""
    print(f"[STEP] Running Streaming RAG Search for query: '{query}'")
    rag = RAGEngine()

    print("\n🔄 Streaming Answer:\n")
    async for token in rag.stream_query(question=query, top_k=3, summarize=True):
        print(token, end="", flush=True)

    print("\n\n✅ [DONE] Streaming complete.")


def main():
    parser = argparse.ArgumentParser(description="Run Basic or Streaming RAG Search.")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="The query/question to ask the RAG system."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["basic", "streaming"], 
        default="basic",
        help="Choose RAG mode: 'basic' or 'streaming'"
    )
    args = parser.parse_args()

    # 1️⃣ Ensure vector store exists
    build_vectorstore()

    # 2️⃣ Run mode-specific logic
    if args.mode == "basic":
        run_basic_rag(args.query)
    else:
        print("\n" + "=" * 60)
        print("🔄 Running in streaming mode...")
        asyncio.run(run_streaming_rag(args.query))


if __name__ == "__main__":
    main()