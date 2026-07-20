import os
import tempfile

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from document_processor import DocumentProcessor
from rag_system import RagSystem
from vector_database import VectorDatabase

st.set_page_config(layout="wide", page_title="AI Requirements Chat", page_icon=":robot:")

@st.cache_resource
def get_embedding_function():
    """Load the embedding model and cache it."""
    if "HF_TOKEN" in st.secrets:
        os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    else:
        print("Warning: No Hugging Face token provided. Add HF_TOKEN=\"hf_key\" to the secrets.toml to remove the warning.")

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.dialog("How to Use the AI Requirements Analyst")
def info_modal():
    st.markdown("""
    ### 📘 How to Use the AI Requirements Analyst

Welcome to your local, AI-powered document analysis platform. This tool acts as an expert software architect, allowing you to instantly search, analyze, and query your project specifications. 

Here is how to get the most out of the platform:

#### 1. 📂 Upload Your Documents
To begin, upload your project files using the **attachment icon** inside the chat box at the bottom of the screen.
* **Supported formats:** `.pdf`, `.docx`, `.csv`
* You can upload multiple files at once. 
* The system will automatically read, chunk, and index your documents into a secure local database.

#### 2. 💬 Ask Highly Specific Questions
Once your files are uploaded, ask questions exactly as you would to a lead engineer. The AI will scan across all uploaded documents to find the answer.
* *Example:* "What does the frontend document say about login timeouts?"
* *Example:* "Are there any contradictions between the API spec and the UI guidelines regarding user roles?"
* *Example:* "Summarize the data migration steps into a bulleted list."

#### 3. ⚙️ How the AI Operates (System Rules)
To ensure absolute accuracy, this AI is bound by strict analytical rules:
* **Zero Hallucination:** The AI will *only* answer based on the documents provided. If the information is missing, it will directly tell you it cannot be found. It will not guess or invent features.
* **Source Citations:** When the AI provides an answer, it will cite the exact document and page number it pulled the information from (e.g., `[Source: SRS_Frontend.pdf | Page: 4]`).
* **Language Matching:** The AI will always respond in the exact same language you used to ask the question. 
*** 

**Ready? Close this window, upload your first document, and say hello!**
""")

# initialize session variables
if "documents" not in st.session_state:
    st.session_state.documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []


if "embedding_function" not in st.session_state:
    st.session_state.embedding_function = get_embedding_function()

if "db" not in st.session_state:
    st.session_state.db = VectorDatabase(st.session_state.embedding_function)

if "rag" not in st.session_state:
    st.session_state.rag = RagSystem(st.session_state.embedding_function)


def add_messages(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def add_document(document: str):
    print(document)
    st.session_state.documents.append(document)

def delete_document(index: int):
    st.session_state.documents.pop(index)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat logic (prompts and documents)
if user_submition := st.chat_input(
        placeholder="What is up?",
        accept_file="multiple",
        file_type=["csv", "pdf", "doc", "docx"],
        submit_mode="disable"
):
    prompt = user_submition.text
    files = user_submition.files

    if files:
        for file in files:
            name = file.name

            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix="." + DocumentProcessor.check_file_extension(name)) as temp_file:
                temp_file.write(file.read())

                file_path = temp_file.name

            add_document(name)

            st.session_state.db.add_documents(file_path)

            os.remove(file_path)

    if prompt:
        add_messages("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.rag.qa_prompt(prompt)
            response = response.content

            st.markdown(response)

        add_messages("assistant", response)



with st.sidebar:
    st.header("AI Model for Analysing Software Requirement")

    st.markdown("Documents from the current session:")

    for index, document in enumerate(st.session_state.documents):
        print(index, document)
        col1, col2 = st.columns([0.8, 0.2])

        with col1:
            st.markdown(f"**{document}**")

        with col2:
            st.button(
                label="",
                icon=":material/delete:",
                on_click=delete_document,
                args=[index]
            )

    with st.container(key="user-guide"):
        if st.button("User Guide", icon=":material/info:"):
            info_modal()

    st.html("""
    <style>
        div[data-testid="stSidebarUserContent"] {
            height: 100%;
        }
        
        div.st-key-user-guide > div {
            position: absolute;
            bottom: 20px;
            right: 20px;
        }
        
    </style>
    """)


if len(st.session_state.messages) == 0:
    st.header("Welcome! I am ready to help you analyze your project documentation.")
    st.markdown("""
**Get started:**
1. Click the `+` (attachment) icon below to upload your files.
2. Wait a moment for the system to process the documents.
3. Ask me any question, like: 'What does the frontend document say about login timeouts?'

What would you like to analyze today?""")