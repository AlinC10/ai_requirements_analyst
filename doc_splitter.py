from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


class DocSplitter:
    """
    Class used for splitting the documents in multiple chunks that will be store into a vector database and feed to
    the LLM to get an answer.
    """

    def __init__(self, extension: str, chunk_size: int = 1000, chunk_overlap: int = 100,
                 separators: list[str] | None = None):
        """
        :param extension: Used to determine if the Markdown can be used for first splitting the document based on the
        Markdown Headers, then after character. (only available for PDF and DOCX documents)
        :type extension: str
        :param chunk_size: Maximum number of characters for a text chunk. 1000 characters by default.
        :type chunk_size: int
        :param chunk_overlap: Maximum number of characters for the overlap between 2 chunks. Default = 100.
        :type chunk_overlap: int
        :param separators: List of characters or Regex expression that will be used for separating the chunks.
        Default = [\"\\n\\n\", \"\\n\", "(?<=[.?!])", " ", ""]
        :type separators: list[str] | None
        """

        if separators is None:
            separators = ["\n\n", "\n", "(?<=[.?!])", " ", ""]

        self.recursive_doc_splitter = DocSplitter.recursive_doc_splitter(chunk_size, chunk_overlap, separators)
        self.extension = extension
        self.markdown_splitter = None

        if self.extension in ['pdf', 'docx']:
            self.markdown_splitter = DocSplitter.markdown_doc_splitter()

    @staticmethod
    def recursive_doc_splitter(chunk_size: int = 1000, chunk_overlap: int = 100,
                               separators: list[str] | None = None) -> RecursiveCharacterTextSplitter:
        """
        Retriever for the splitter used for every file with or without the Markdown splitter.

        :param chunk_size: Maximum number of characters for a text chunk. 1000 characters by default.
        :type chunk_size: int
        :param chunk_overlap: Maximum number of characters for the overlap between 2 chunks. Default = 100.
        :type chunk_overlap: int
        :param separators: List of characters or Regex expression that will be used for separating the chunks.
        Default = [\"\\n\\n\", \"\\n\", "(?<=[.?!])", " ", ""]
        :type separators: list[str] | None
        :return: Splitter used for every file, based on number of characters.
        :rtype: RecursiveCharacterTextSplitter
        """

        if separators is None:
            separators = ["\n\n", "\n", "(?<=[.?!])", " ", ""]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True
        )

        return splitter

    @staticmethod
    def markdown_doc_splitter() -> MarkdownHeaderTextSplitter:
        """
        Retriever for the splitter used for the Markdown.

        :return: Splitter that will be used for Markdowns.
        :rtype: MarkdownHeaderTextSplitter
        """

        # where the markdown splitter will divide the text
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
        """
        Split documents for the RAG system based on their type.
        If the file send to the system is PDF or DOCX, it's first divided into Markdown sections, then after
        sections numbers of characters using RecursiveCharacterTextSplitter.
        If the file send is a TXT, it will be divided using only RecursiveCharacterTextSplitter.

        :param document: The content of the document which will be split using the rules from above.
        :type document: Document
        :return: Document divided into multiple chunks using the rules from above.
        :rtype: list[Document]
        """

        content = document.page_content
        metadata = document.metadata

        if self.markdown_splitter is not None:
            docs = self.markdown_splitter.split_text(content)
        else:
            docs = [document]

        for doc in docs:
            doc.metadata.update(metadata)

        recursive_split_docs = self.recursive_doc_splitter.split_documents(docs)

        return recursive_split_docs
