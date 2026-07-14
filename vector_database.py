import os
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from document_processor import DocumentProcessor

load_dotenv()


class VectorDatabase:
    def __init__(self):
        # Safely gets the env variable, defaults to './chroma_db' if missing
        self._persistent_directory = os.environ.get("CHROMA_DIR", "./chroma_db")

        # Use a local HuggingFace embedding model.
        # The `sentence-transformers` library will automatically download this model from
        # the Hugging Face Hub the first time it's used and cache it locally.
        self._embeddings_function = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

        self.collection = self.get_or_create_collection()
        self.dp = DocumentProcessor()

    def get_or_create_collection(self, collection_name: str = "collection") -> Chroma:
        collection = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings_function,
            persist_directory=self._persistent_directory
        )
        return collection

    # def add_documents(self, docs: list[Document]) -> None:
    #     conversation_id = f"chat_{int(time.time())}"
    #
    #     for doc in docs:
    #         doc.metadata["conversation_id"] = conversation_id
    #
    #     self.collection.add_documents(docs)

    def add_documents(self, docs_file_path: str) -> None:
        conversation_id = f"chat_{int(time.time())}"

        docs = self.dp.load_doc(docs_file_path, conversation_id)

        self.collection.add_documents(docs)

    def get_retriever(self, search_kwargs: dict | None = None, k: int = 5) -> VectorStoreRetriever:
        """
        Create a retriever to get most similar k chunks from the Chroma collection.
        :param k: number of chunks to be retrieved
        :param search_kwargs: filters used to get select information from Chroma database

        :return retriever: search database and returns the relevant chunks
        """

        if search_kwargs is None:
            search_kwargs = {}

        search_kwargs['k'] = k

        retriever = self.collection.as_retriever(
            search_type='similarity',
            search_kwargs=search_kwargs
        )

        return retriever

    def retrieve_data(self, prompt: str, search_kwargs: dict | None = None, k: int = 5) -> list[Document]:
        """
        Search for the most similar k elements from the database using a retriever.

        :param prompt: user prompt used for searching the database
        :param search_kwargs: filters used to get select information from Chroma database
        :param k: number of chunks to be retrieved

        :return relevant chunks: the k chunks most similar with the prompt from the database
        """

        retriever = self.get_retriever(search_kwargs, k)

        relevant_chunks = retriever.invoke(prompt)

        return relevant_chunks
