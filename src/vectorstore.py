import os
import uuid
import chromadb
from typing import List, Any
from src.embedding import EmbeddingPipeline

class ChromaVectorStore:
    def __init__(self,
                 collection_name: str = "pdf_documents",
                 persist_directory: str = "chromadb_store",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        # Initialize embedding pipeline
        self.emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Initialize Chroma client and collection
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF embeddings using Azure OpenAI"}
        )

        print(f"[INFO] Chroma vector store initialized: {self.collection_name}")
        print(f"[INFO] Existing documents in collection: {self.collection.count()}")

    def add_documents(self, documents: List[Any]):
        """
        Chunks, embeds, and adds documents to ChromaDB.
        """
        print(f"[INFO] Adding {len(documents)} documents to Chroma store...")

        # Split documents into chunks
        chunks = self.emb_pipe.chunk_documents(documents)

        # Generate embeddings using Azure OpenAI
        embeddings = self.emb_pipe.generate_embeddings(chunks)

        ids, metadatas = [], []

        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = {
                "doc_index": i,
                "content_length": len(chunk_text)
            }
            metadatas.append(metadata)

        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=chunks
        )
        print(f"[INFO] Successfully added {len(chunks)} chunks.")
        print(f"[INFO] Total documents in collection: {self.collection.count()}")

    def query(self, query_text: str, top_k: int = 5):
        """
        Query ChromaDB using Azure embeddings.
        """
        query_emb = self.emb_pipe.generate_embeddings([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def list_documents(self):
        return self.collection.get(include=["metadatas", "documents"])

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
        print(f"[INFO] Collection '{self.collection_name}' deleted.")


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents
    docs = load_all_documents("data")

    store = ChromaVectorStore(persist_directory="chromadb_store")
    store.add_documents(docs)

    query_result = store.query("Explain attention mechanism", top_k=3)
    for doc, meta, dist in zip(query_result['documents'][0],
                               query_result['metadatas'][0],
                               query_result['distances'][0]):
        print(f"Distance: {dist:.4f}, Text snippet: {doc[:150]}...")