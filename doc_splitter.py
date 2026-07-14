from langchain_text_splitters import RecursiveCharacterTextSplitter


def doc_splitter(chunk_size: int = 1000, chunk_overlap: int = 100,
                 separators: list[str] = ["\n\n", "\n", "(?<=[.?!])", " ", ""]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=True,
        keep_separator=True
    )

    return splitter
