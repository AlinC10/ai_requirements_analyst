import os.path

import docx
# from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from markitdown import MarkItDown
from pymupdf4llm.ocr import rapidocr_api

from doc_splitter import DocSplitter

default_important_metadata_categories: list[str] = ['title', 'author', 'subject', 'keywords', 'source']
page_break_delimiter = "\n---PAGE BREAK---\n"


def split_doc(docs: Document, extension: str, chunk_size: int = 1000, chunk_overlap: int = 100,
              separator: list[str] | None = None) -> list[Document]:
    """Split the documents based on the rules defined in the DocSplitter object."""

    splitter = DocSplitter(extension, chunk_size, chunk_overlap, separator)

    split_docs = splitter.document_split(docs)

    return split_docs


def check_file_extension(file_path: str):
    _, extension = os.path.splitext(file_path)
    extension = extension[1:]

    accepted_types = ["docx", "pdf", "txt"]

    if extension in accepted_types:
        return extension

    raise TypeError(f"Accepted document types: {", ".join(accepted_types)}")


def _remove_useless_info_from_metadata(docs: Document,
                                       important_metadata_categories: list[str] | None = None) -> Document:
    """Remove useless information from metadata"""

    if important_metadata_categories is None:
        important_metadata_categories = default_important_metadata_categories

    keys_to_delete = [key for key in docs.metadata if key not in important_metadata_categories]

    for key in keys_to_delete:
        del docs.metadata[key]

    return docs


def _add_document_metadata(docs: list[Document], chat_id: str, extension: str):
    """Add chat id attribute to the document metadata for better filtering in the database collection.
    If the document added it's a PDF, it will also add the page number into the metadata.
    """

    page_number = None

    if extension == "pdf":
        page_number = 1

    for doc in docs:
        doc.metadata['chat_id'] = chat_id

        if page_number:
            if page_break_delimiter in doc.page_content:
                page_number += 1

            doc.metadata['page'] = page_number

    print(docs[0].metadata)
    return docs


def load_doc(file_path: str, chat_id: str, file_name: str,
             important_metadata_categories: list[str] | None = None) -> list[Document]:
    if important_metadata_categories is None:
        important_metadata_categories = default_important_metadata_categories

    # Checks file extension
    extension = check_file_extension(file_path)

    docs = None
    if extension in 'docx':
        docs = _load_docx(file_path, file_name, chat_id)
    elif extension == "pdf":
        docs = _load_pdf(file_path, file_name, chat_id)
        docs = _remove_useless_info_from_metadata(docs, important_metadata_categories)
    else:
        docs = _load_txt(file_path, file_name, chat_id)

    split_docs = split_doc(docs, extension)

    return _add_document_metadata(split_docs, file_path, chat_id)


def extract_file_name(file_path: str) -> str:
    """Extract the fie name from a given file path.
    e.g.: pdf/file_name.pdf = file_name
    """

    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]

    return name_without_ext


def extract_file_title(file_path: str, text: str = "", meta_title: str | None = None):
    """Finds the best possible title using a fallback chain."""

    if meta_title and meta_title.strip() not in ["", "Untitled"]:
        return meta_title

    # search for "# " which is used for titles in Markdown format
    for line in text.splitlines():
        line = line.strip()
        print('1 ' + line)
        if line.startswith('# '):
            print("2 " + line)
            return line[2:].strip(" *")

    # create the title from file path, removing "_" and "-" symbols
    # e.g.: pdf/file_name.pdf = File Name
    file_name_without_ext = extract_file_name(file_path)
    clean_name = file_name_without_ext.replace("_", " ").replace("-", " ").title()

    return clean_name


def _load_docx(file_path: str, file_name: str, chat_id: str) -> Document:
    """Load the .docx file using MarkItDown library."""

    md = MarkItDown()
    document = md.convert(file_path).text_content

    # load .docx properties
    doc = docx.Document(file_path)
    props = doc.core_properties

    # extract the text of the first paragraph styled as a 'Title' from .docx to retrieve the document title
    title = None
    try:
        for para in doc.paragraphs[:5]:
            if para.style.name.startswith("Title") and para.text.strip():
                title = para.text.strip()
                break
    except Exception:
        pass

    metadata = {"author": props.author or "Unknown",
                "title": title or extract_file_title(file_name, document, props.title),
                "category": props.category or "Unknown",
                "source": file_name,
                "chat_id": chat_id
                }

    return Document(
        page_content=document,
        metadata=metadata
    )


def _load_txt(file_path: str, file_name: str, chat_id: str) -> Document:
    """Load the .txt file.
    If Python encounters a corrupted or weird character that isn't valid UTF-8, instead of crashing the entire
    script with a UnicodeDecodeError, it simply replaces the bad character with a standard placeholder symbol (
    usually ``).
    """
    with open(file_path, mode="r", encoding="utf-8", errors="replace") as file:
        txt = file.read()
        return Document(page_content=txt,
                        metadata={"source": file_name, "title": extract_file_title(file_name), "chat_id": chat_id})


def _load_pdf(file_path: str, file_name: str, chat_id: str) -> Document:
    """Load the PDF file into the RAM using PyMuPDF4LLMLoader with
    RapidOCR for retrieving text from images."""
    loader = PyMuPDF4LLMLoader(
        file_path=file_path,
        mode="page",
        pages_delimiter=page_break_delimiter,
        # extract_images=True,
        extract_images=False,

        # use RapidOCR for extracted image fragments
        # images_parser=RapidOCRBlobParser(),

        # tell PyMuPDF4LLM to use RapidOCR instead of Tesseract for scanned pages
        ocr_function=rapidocr_api.exec_ocr
    )

    document = loader.load()[0]
    document.metadata.update({"chat_id": chat_id, "source": file_name})

    title = extract_file_title(file_path, file_name, document.metadata.get('title', None))
    document.metadata.update({'title': title})

    return document


def format_for_llm(docs: list[Document]) -> str:
    formated_context = "\n\n".join(
        f"""[Source: {doc.metadata.get('title', 'Unknown')} 
{f"| Page: {doc.metadata['page']}" if doc.metadata.get("page", False) else ""}]\n
{doc.page_content}"""
        for doc in docs
    )

    return formated_context
