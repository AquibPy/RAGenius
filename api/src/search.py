import os
import asyncio
import logging
from typing import Dict, Any, AsyncGenerator, Optional, List
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from api.src.vectorstore import ChromaVectorStore
from api.src.data_loader import load_all_documents

load_dotenv()

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Unified Retrieval-Augmented Generation engine using Azure OpenAI + ChromaDB.
    Supports both normal and streaming query modes.
    """

    def __init__(
        self,
        persist_dir: str = "chromadb_store",
        llm_provider="azure",      
        embedding_provider="azure", 
        llm_model: str = "gpt-4",
        temperature: float = 0.7,
        streaming: bool = True,
    ):
        
        self.llm_provider = llm_provider.lower()
        self.embedding_provider = embedding_provider.lower()

        self.vectorstore = ChromaVectorStore(
                persist_directory=persist_dir,
                # embedding_model=self.embedding_provider,
                provider=self.embedding_provider
            )
        
        if self.llm_provider=="azure":
            # Initialize Azure Chat LLM
            self.llm = AzureChatOpenAI(
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=temperature,
                model_name=llm_model,
                streaming=streaming
            )
        elif self.llm_provider=="groq":
            logger.info("[LLM] Using Groq model")
            from langchain_groq import ChatGroq

            self.llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model="openai/gpt-oss-120b",
                temperature=0.7,
                max_retries=3,
                streaming=streaming,
                reasoning_format="parsed"
                )
        else:
            raise ValueError(f"Invalid provider: {self.llm_provider}. Use 'azure' or 'groq'.")
        
        # Ensure vectorstore is populated
        self._ensure_documents_loaded()


        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # Internal Utility Methods
    # ------------------------------------------------------------
    def _ensure_documents_loaded(self):
        """Load documents only if the vector store is empty."""
        count = self.vectorstore.collection.count()
        if count == 0:
            logger.info("No documents found in vector store. Loading and adding documents...")
            docs = load_all_documents("data")
            self.vectorstore.add_documents(docs)
            logger.info(f"Added {len(docs)} new documents to vector store.")
        else:
            logger.info(f"Vector store already contains {count} documents.")

    def _build_prompt(self, context: str, question: str, summarize: bool) -> str:
        """Constructs the LLM prompt."""
        base_prompt = f"""
        You are a helpful assistant. Use the following context to answer the user's question.

        Context:
        {context}

        Question: {question}
        """
        if summarize:
            base_prompt += "\nAlso, provide a 2-sentence summary at the end of your answer."
        return base_prompt

    # ------------------------------------------------------------
    # Core Query Functions
    # ------------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: int = 5,
        summarize: bool = False
    ) -> Dict[str, Any]:
        """Retrieves context and gets LLM-generated response (non-streaming)."""
        try:
            results = self.vectorstore.query_db(question, top_k=top_k)
            docs = results.get("documents", [[]])[0]

            if not docs:
                logger.warning("No relevant context found.")
                return {"answer": "No relevant context found.", "sources": [], "history": self.history}

            context = "\n\n".join(docs)
            prompt = self._build_prompt(context, question, summarize)

            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            answer = response.content

            self.history.append({"question": question, "answer": answer})
            return {
                "question": question,
                "answer": answer,
                "sources": results.get("metadatas", []),
                "history": self.history
            }

        except Exception as e:
            logger.error(f"Error during query: {e}", exc_info=True)
            return {"error": str(e)}

    # ------------------------------------------------------------
    # Streaming Version
    # ------------------------------------------------------------
    async def stream_query(
        self,
        question: str,
        top_k: int = 5,
        summarize: bool = False
    ) -> AsyncGenerator[str, None]:
        """Async generator for true token-by-token streaming output."""
        try:
            results = self.vectorstore.query_db(question, top_k=top_k)
            docs = results.get("documents", [[]])[0]

            if not docs:
                yield "No relevant context found."
                return

            context = "\n\n".join(docs)
            prompt = self._build_prompt(context, question, summarize)

            yield "Answer:\n"
            answer = ""

            async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
                token = getattr(chunk, "content", str(chunk))
                answer += token
                yield token

            self.history.append({"question": question, "answer": answer})

        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            yield f"[Error] {str(e)}"

if __name__ == "__main__":
    rag = RAGEngine(llm_provider="azure",embedding_provider="gemini")

    # Normal Query
    result = rag.query("What is the attention mechanism?", top_k=3, summarize=True)
    print("\nAnswer:\n", result["answer"])
    # print(result)

    # Streaming Example
    # async def main():
    #     async for token in rag.stream_query("Explain transformers in NLP", top_k=3):
    #         print(token, end="", flush=True)

    # asyncio.run(main())