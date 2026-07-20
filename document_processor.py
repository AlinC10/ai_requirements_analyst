import os.path

from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from pymupdf4llm.ocr import rapidocr_api

from doc_splitter import doc_splitter

default_important_metadata_categories: list[str] = ['title', 'author', 'subject', 'keywords', 'page']


class DocumentProcessor:
    def __init__(self):
        pass

    @staticmethod
    def split_doc(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> list[Document]:
        splitter = doc_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splitted_docs = splitter.split_documents(docs)

        return splitted_docs

    @staticmethod
    def check_file_extension(file_path: str):
        _, extension = os.path.splitext(file_path)
        extension = extension[1:]

        accepted_types = ["doc", "docx", "pdf", "txt"]

        if extension in accepted_types:
            return extension

        raise TypeError(f"Accepted document types: {", ".join(accepted_types)}")

    @staticmethod
    def _remove_useless_info_from_metadata(docs: list[Document],
                                           important_metadata_categories: list[str] = default_important_metadata_categories) -> \
            list[Document]:
        """Remove useless information from metadata"""
        for doc in docs:
            keys_to_delete = [key for key in doc.metadata if key not in important_metadata_categories]

            for key in keys_to_delete:
                del doc.metadata[key]

        return docs

    @staticmethod
    def _add_document_metadata(docs: list[Document], file_path: str, chat_id: str):
        file_name = os.path.basename(file_path)
        for doc in docs:
            doc.metadata['source'] = file_name
            doc.metadata['conversation_id'] = chat_id

        return docs

    @staticmethod
    def load_doc(file_path: str, chat_id: str, file_name: str | None = None,
                 important_metadata_categories: list[str] = default_important_metadata_categories) -> list[Document]:

        """Checks file extension"""
        extension = DocumentProcessor.check_file_extension(file_path)

        docs = None
        if extension in ['doc', 'docx']:
            docs = DocumentProcessor._load_docx(file_path)
        elif extension == "pdf":
            docs = DocumentProcessor._load_pdf(file_path)
        else:
            docs = DocumentProcessor._load_txt(file_path)

        splitted_docs = DocumentProcessor.split_doc(docs)

        splitted_docs = DocumentProcessor._remove_useless_info_from_metadata(splitted_docs,
                                                                             important_metadata_categories)

        return DocumentProcessor._add_document_metadata(splitted_docs, file_path, chat_id)

    @staticmethod
    def _load_docx(file_path: str):
        pass

    @staticmethod
    def _load_txt(file_path: str):
        pass

    @staticmethod
    def _load_pdf(file_path: str) -> list[Document]:
        """Load the PDF file into the RAM using PyMuPDF4LLMLoader with
        RapidOCR for retrieving text from images."""
        loader = PyMuPDF4LLMLoader(
            file_path=file_path,
            mode="page",
            extract_images=True,

            # use RapidOCR for extracted image fragments
            images_parser=RapidOCRBlobParser(),

            # tell PyMuPDF4LLM to use RapidOCR instead of Tesseract for scanned pages
            ocr_function=rapidocr_api.exec_ocr
        )

        return loader.load()

    @staticmethod
    def format_for_llm(docs: list[Document]) -> str:
        formated_context = "\n\n".join(
            f"""[Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"""
            for doc in docs
        )

        return formated_context

    # @staticmethod
    # def process_docs(file_path:str, chunk_size: int = 1000, chunk_overlap: int = 100, important_metadata_categories: list[str]=default_important_metadata_categories) -> list[Document]:
    #     docs = DocumentProcessor.load_doc(file_path ,important_metadata_categories)
    #
    #     splitted_docs = DocumentProcessor.split_doc(docs, chunk_size, chunk_overlap)
    #
    #     return splitted_docs
