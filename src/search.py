# import os
# from dotenv import load_dotenv
# from langchain_openai import AzureChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage
# from src.vectorstore import ChromaVectorStore  # your Chroma wrapper
# from src.data_loader import load_all_documents
# from typing import Dict, Any, AsyncGenerator
# import asyncio

# load_dotenv()

# class RAGSearch:
#     def __init__(self, 
#                  persist_dir: str = "chromadb_store",
#                  embedding_model: str = None):

#         self.vectorstore = ChromaVectorStore(
#             persist_directory=persist_dir,
#             embedding_model=embedding_model
#         )

#         if self.vectorstore.collection.count() == 0:
#             print("[INFO] No documents in vector store. Loading and adding documents...")
#             docs = load_all_documents("data")
#             self.vectorstore.add_documents(docs)
#         else:
#             print(f"[INFO] Vector store already has {self.vectorstore.collection.count()} documents.")

#         self.llm = AzureChatOpenAI(
#             openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#             azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#             api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#             temperature=0.7
#         )

#     def search_and_summarize(self, query: str, top_k: int = 5) -> str:
#         """
#         Retrieves top_k relevant documents from ChromaDB and summarizes them using Azure Chat LLM.
#         """
#         results = self.vectorstore.query(query, top_k=top_k)

#         texts = results['documents'][0] if results['documents'] else []

#         context = "\n\n".join(texts)

#         if not context:
#             return "No relevant documents found."

#         prompt = f"""
#         You are a helpful assistant. Summarize the following context for the query: '{query}'.

#         Context:
#         {context}

#         Summary:
#         """

#         messages = [
#             SystemMessage(content="You are a helpful assistant."),
#             HumanMessage(content=prompt)
#         ]

#         response = self.llm.invoke(messages)
#         return response.content


# class RAGSearchStream:
#     def __init__(self,
#                  persist_dir: str = "chromadb_store",
#                  embedding_model: str = None,
#                  llm_model: str = "gpt-4",
#                  temperature: float = 0.7,
#                  streaming: bool = True):

#         self.vectorstore = ChromaVectorStore(
#             persist_directory=persist_dir,
#             embedding_model=embedding_model
#         )

#         if self.vectorstore.collection.count() == 0:
#             print("[INFO] No documents in vector store. Loading and adding documents...")
#             docs = load_all_documents("data")
#             self.vectorstore.add_documents(docs)
#         else:
#             print(f"[INFO] Vector store already has {self.vectorstore.collection.count()} documents.")

#         self.llm = AzureChatOpenAI(
#             openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#             azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#             api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#             temperature=temperature,
#             model_name=llm_model,
#             streaming=streaming
#         )

#         self.history = []

#     def _build_prompt(self, context: str, question: str) -> str:
#         return f"""
#                     Use the following context to answer the question concisely.

#                     Context:
#                     {context}

#                     Question: {question}

#                     Answer:
#                 """

#     def query(self,
#               question: str,
#               top_k: int = 5,
#               stream: bool = False,
#               summarize: bool = False) -> Dict[str, Any]:
#         """Non-streaming mode"""
#         results = self.vectorstore.query(question, top_k=top_k)
#         docs = results['documents'][0] if results['documents'] else []

#         if not docs:
#             return {"answer": "No relevant context found.", "summary": None, "sources": [], "history": self.history}

#         context = "\n\n".join(docs)
#         prompt = self._build_prompt(context, question)

#         response = self.llm.generate([[HumanMessage(content=prompt)]])
#         answer = response.generations[0][0].text

#         summary_text = None
#         if summarize and answer:
#             summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
#             summary_resp = self.llm.generate([[HumanMessage(content=summary_prompt)]])
#             summary_text = summary_resp.generations[0][0].text

#         self.history.append({"question": question, "answer": answer, "summary": summary_text})

#         return {
#             "question": question,
#             "answer": answer,
#             "summary": summary_text,
#             "history": self.history
#         }
#     ## This fucntion is designed especially for FastAPI
#     async def stream_query(self,
#                            question: str,
#                            top_k: int = 5,
#                            summarize: bool = False) -> AsyncGenerator[str, None]:
#         """Async generator version for true token streaming"""
#         results = self.vectorstore.query(question, top_k=top_k)
#         docs = results['documents'][0] if results['documents'] else []

#         if not docs:
#             yield "No relevant context found."
#             return

#         context = "\n\n".join(docs)
#         prompt = self._build_prompt(context, question)

#         answer = ""
#         async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
#             token = getattr(chunk, "content", str(chunk))
#             answer += token
#             yield token
#             await asyncio.sleep(0.001)

#         self.history.append({"question": question, "answer": answer})



# # Example usage
# if __name__ == "__main__":
#     # rag_search = RAGSearch()
#     # query = "What is attention mechanism?"
#     # summary = rag_search.search_and_summarize(query, top_k=3)
#     # print("Summary:\n", summary)
#     rag_search = RAGSearchStream()
#     result = rag_search.query(
#         "What is attention mechanism?",
#         top_k=3,
#         stream=True,
#         summarize=True
#     )

#     print("\nFinal Answer:\n", result['answer'])
#     print("Summary:\n", result['summary'])
#     print("History (last query):\n", result['history'][-1])


import os
import asyncio
import logging
from typing import Dict, Any, AsyncGenerator, Optional, List
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.vectorstore import ChromaVectorStore
from src.data_loader import load_all_documents

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
        embedding_model: Optional[str] = None,
        llm_model: str = "gpt-4",
        temperature: float = 0.7,
        streaming: bool = True
    ):
        # Initialize Vector Store
        self.vectorstore = ChromaVectorStore(
            persist_directory=persist_dir,
            embedding_model=embedding_model
        )

        # Ensure vectorstore is populated
        self._ensure_documents_loaded()

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
            results = self.vectorstore.query(question, top_k=top_k)
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
            results = self.vectorstore.query(question, top_k=top_k)
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
    rag = RAGEngine()

    # Normal Query
    result = rag.query("What is the attention mechanism?", top_k=3, summarize=True)
    print("\nAnswer:\n", result["answer"])

    # Streaming Example
    async def main():
        async for token in rag.stream_query("Explain transformers in NLP", top_k=3):
            print(token, end="", flush=True)

    asyncio.run(main())