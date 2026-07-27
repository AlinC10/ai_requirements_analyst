import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from streamlit import secrets

import document_processor as dp


class VectorDatabase:
    def __init__(self, embedding_function: Embeddings):
        # Safely gets the env variable, defaults to './chroma_db' if missing
        self._persistent_directory = secrets["CHROMA_DIR"] or "./chroma_db"
        self._embeddings_function = embedding_function
        self.collection = self.get_or_create_collection()

    def get_or_create_collection(self, collection_name: str = "collection") -> Chroma:
        collection = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings_function,
            persist_directory=self._persistent_directory
        )
        return collection

    def add_documents(self, docs_file_path: str, file_name: str) -> None:
        conversation_id = f"chat_{int(time.time())}"
        docs = dp.load_doc(docs_file_path, conversation_id, file_name)
        self.collection.add_documents(docs)

    def delete_documents(self, source: str) -> None:
        self.collection.delete(where={"source": source})

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
