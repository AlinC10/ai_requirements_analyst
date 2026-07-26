from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


class DocSplitter:
    def __init__(self, extension: str, chunk_size: int = 1000, chunk_overlap: int = 100,
                               separators: list[str] | None = None):
        if separators is None:
            separators = ["\n\n", "\n", "(?<=[.?!])", " ", ""]

        self.recursive_doc_splitter = DocSplitter.recursive_doc_splitter(chunk_size, chunk_overlap, separators)
        self.extension = extension
        self.markdown_splitter = None

        if self.extension in ['pdf', 'docx']:
            self.markdown_splitter = DocSplitter.markdown_doc_splitter()

    @staticmethod
    def recursive_doc_splitter(chunk_size: int = 1000, chunk_overlap: int = 100,
                               separators: list[str] | None = None):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True
        )

        return splitter

    @staticmethod
    def markdown_doc_splitter():
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        return splitter

    def document_split(self, document: Document) -> list[Document]:
        """Split documents for the RAG system based on their type.
        If the file send to the system is PDF or DOCX, it's first divided into Markdown sections, then after
        sections numbers of characters using RecursiveCharacterTextSplitter.
        If the file send is a TXT, it will be divided using only RecursiveCharacterTextSplitter.
        """
        content = document.page_content
        metadata = document.metadata

        docs = None

        if self.markdown_splitter is not None:
            docs = self.markdown_splitter.split_text(content)
        else:
            docs = [document]

        for doc in docs:
            doc.metadata.update(metadata)

        recursive_split_docs = self.recursive_doc_splitter.split_documents(docs)

        return recursive_split_docs