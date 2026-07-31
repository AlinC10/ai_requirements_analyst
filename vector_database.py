import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from streamlit import secrets

import document_processor as dp


class VectorDatabase:
    """
    Class used to control the Chroma Vector Database that will store documents chunks as vectors, that helps retrieve
    chunks based on the similarity with the user message.
    """

    def __init__(self, embedding_function: Embeddings, collection_name: str = "collection"):
        """
        Get a collection in the database that will use an embedding function needed to map text to a vector (a
        point in n-dimensional space).

        :param embedding_function: Function used to map text to a vector (a point in n-dimensional space).
        :type embedding_function: Embeddings
        :param collection_name: Name used for the collection. Default = \"collection\".
        :type collection_name: str
        """

        # Safely gets the env variable, defaults to './chroma_db' if missing
        self._persistent_directory = secrets["CHROMA_DIR"] or "./chroma_db"
        self._embeddings_function = embedding_function
        self.collection = self.get_or_create_collection(collection_name)

    def get_or_create_collection(self, collection_name: str = "collection") -> Chroma:
        """
        Create (if it does not already exist) and connect to the Chroma Database collection.

        :param collection_name: Name used for the collection. Default = \"collection\"
        :type collection_name: str
        :return: Chroma Database collection obtained.
        :rtype: Chroma
        """

        collection = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings_function,
            persist_directory=self._persistent_directory
        )
        return collection

    def add_documents(self, file_path: str, file_name: str) -> None:
        """
        Add a document into the database by using his file path for loading the document and file name as source
        name. It adds a chat id for better retrieval when the chat conversation history will be added.

        :param file_path: Used for loading the document. Cannot be used as file name because this is the temporary
        file path, created when it was added in the UI by the user and will not help the citations if it's used as
        file name.
        :type file_path: str
        :param file_name: Original file name, used as source name.
        :type file_name: str
        :return: None.
        :rtype: None
        """

        chat_id = f"chat_{int(time.time())}"
        docs = dp.load_doc(file_path, chat_id, file_name)
        self.collection.add_documents(docs)

    def delete_documents(self, source: str) -> None:
        """
        Delete all the document chunks from the databased based on the document source metadata.

        :param source: File location extracted from the metadata, used for deleting all the chunks from that document.
        :type source: str
        :return: None.
        :rtype: None
        """

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
        Search for the most similar k filtered chunks from the database using a retriever.

        :param prompt: user prompt used for searching the database
        :param search_kwargs: filters used to get select information from Chroma database
        :param k: number of chunks to be retrieved

        :return relevant chunks: the k chunks most similar with the prompt from the database
        """
        retriever = self.get_retriever(search_kwargs, k)
        relevant_chunks = retriever.invoke(prompt)
        return relevant_chunks
