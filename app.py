import os
import shutil
import subprocess
import tempfile
import time
from typing import Any

import ollama
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

import document_processor as dp
import prompts
from rag_system import RagSystem

st.set_page_config(layout="wide", page_title="AI Requirements Chat", page_icon=":robot:")


@st.cache_resource
def get_embedding_function():
    """Load the embedding model and cache it."""
    if "HF_TOKEN" in st.secrets:
        os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    else:
        print(
            "Warning: No Hugging Face token provided. Add HF_TOKEN=\"hf_key\" to the secrets.toml to remove the "
            "warning.")

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.dialog("How to Use the AI Requirements Analyst")
def info_modal():
    st.markdown("""
### 📘 How to Use the AI Requirements Analyst

Welcome to your local, AI-powered document analysis platform. This tool acts as an expert software architect, 
allowing you to instantly search, analyze, and query your project specifications. 

Here is how to get the most out of the platform:

#### 1. 📂 Upload Your Documents
To begin, upload your project files using the **attachment icon** inside the chat box at the bottom of the screen.
* **Supported formats:** `.pdf`, `.docx`, `.csv`
* You can upload multiple files at once. 
* The system will automatically read, chunk, and index your documents into a secure local database.

#### 2. 💬 Ask Highly Specific Questions
Once your files are uploaded, ask questions exactly as you would to a lead engineer. The AI will scan across all 
uploaded documents to find the answer.
* *Example:* "What does the frontend document say about login timeouts?"
* *Example:* "Are there any contradictions between the API spec and the UI guidelines regarding user roles?"
* *Example:* "Summarize the data migration steps into a bulleted list."

#### 3. ⚙️ How the AI Operates (System Rules)
To ensure absolute accuracy, this AI is bound by strict analytical rules:
* **Zero Hallucination:** The AI will *only* answer based on the documents provided. If the information is missing, 
it will directly tell you it cannot be found. It will not guess or invent features.
* **Source Citations:** When the AI provides an answer, it will cite the exact document and page number it pulled the 
information from (e.g., `[Source: SRS_Frontend.pdf | Page: 4]`).
* **Language Matching:** The AI will always respond in the exact same language you used to ask the question. 
*** 

**Ready? Close this window, upload your first document, and say hello!**
""")


def add_messages(role: str, content: str, command: str | None = None, metadata: dict[str, Any] | None = None) -> None:
    message = {"role": role, "command": command, "content": content}

    if metadata is not None:
        message.update({"metadata": metadata})

    st.session_state.messages.append(message)


def show_welcome_msg() -> None:
    message = """
    ## Welcome! I am ready to help you analyze your project documentation.
    **Get started:**
    1. Click the `+` (attachment) icon below to upload your files.
    2. Wait a moment for the system to process the documents.
    3. Ask me any question, like: 'What does the frontend document say about login timeouts?'

    What would you like to analyze today?"""

    add_messages('assistant', message)


# initialize session variables
if "documents" not in st.session_state:
    st.session_state.documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "embedding_function" not in st.session_state:
    st.session_state.embedding_function = get_embedding_function()

if "rag" not in st.session_state:
    show_welcome_msg()

    # deletes Chroma DB if it exists from previous
    chroma_db_path = st.secrets["CHROMA_DIR"] or "chroma_db"

    st.session_state.rag = RagSystem(st.session_state.embedding_function)

if "db" not in st.session_state:
    st.session_state.db = st.session_state.rag.vector_database
    st.session_state.db.collection.reset_collection()

if "llm_mode" not in st.session_state:
    st.session_state.llm_mode = "Cloud" # Initialize with a default value


def add_document(document: str, file_path: str) -> None:
    st.session_state.documents.append({'source': document, 'isActive': True})
    st.session_state.db.add_documents(file_path, document)


def modify_document_state(index: int, checkbox_key: str) -> None:
    current_state = st.session_state.get(checkbox_key, st.session_state.documents[index].get('isActive', True))
    st.session_state.documents[index]['isActive'] = current_state


def delete_document(index: int) -> None:
    source = st.session_state.documents.pop(index)['source']
    st.session_state.db.delete_documents(source)


def show_mermaid_diagram(code: str) -> None:
    st.mermaid_chart(code)

    with st.expander("View Mermaid Code"):
        st.code(code, language="Mermaid")


def show_chat(message: dict[str, Any]):
    command = message.get("command", None)
    content = message["content"]

    if command == "diagram":
        show_mermaid_diagram(content)
    elif command == "success":
        st.success(content)
    else:
        st.markdown(content)

    metadata = message.get("metadata", None)
    if metadata is not None:
        st.html(f'''
        <div class="response-metadata">
            <p>Response given in <strong>{metadata["total_time"]}s</strong>, using <strong>
{metadata["model_name"]}</strong>. Token Usage: {metadata["input_tokens"]} (prompt) + {metadata["output_tokens"]} (
response) = <strong>{metadata["total_tokens"]}</strong> tokens.</p> 
        </div>''')


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        show_chat(message)

# Chat logic (prompts and documents)
if user_submission := st.chat_input(
        placeholder="What is up?",
        accept_file="multiple",
        file_type=["txt", "pdf", "docx"],
        submit_mode="disable"
):
    prompt = (user_submission.text or "").strip()
    files = user_submission.files

    if files:
        with st.chat_message("assistant"):
            label = "Loading the " + ("documents" if len(files) > 1 else "document")
            with st.spinner(label, show_time=True):
                for file in files:
                    # start the timer for every document loading
                    start_time = time.time()

                    name = file.name
                    with tempfile.NamedTemporaryFile(delete=False,
                                                     suffix="." + dp.check_file_extension(name)) as temp_file:
                        temp_file.write(file.read())

                        file_path = temp_file.name

                    add_document(name, file_path)

                    os.remove(file_path)

                    # stop the timer for every document loaded
                    stop_time = time.time()
                    # get elapsed time and round it to 2 decimal places
                    elapsed_time = round(stop_time - start_time, 2)

                    content = f"**{name}** was loaded in **{elapsed_time}s**."
                    add_messages("assistant", content, "success")
                    st.success(content)

    if prompt:
        add_messages("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Wait for AI to respond...", show_time=True):
                start_time = time.time()
                command, prompt = prompts.retrieve_command(prompt)

                complementary_system_prompt = prompts.get_specific_system_prompt(command)

                active_documents = []
                inactive_documents = []

                for document in st.session_state.documents:
                    if document['isActive']:
                        active_documents.append(document['source'])
                    else:
                        inactive_documents.append(document['source'])

                search_kwargs = None

                props_deleted = 0
                if len(active_documents) != 0:
                    search_kwargs = {
                        'filter': {
                            'source': {
                                '$in': active_documents,
                            }
                        }
                    }
                elif len(inactive_documents) != 0:
                    search_kwargs = {
                        'filter': {
                            'source': {
                                '$nin': inactive_documents,
                            }
                        }
                    }

                response = st.session_state.rag.get_response(complementary_system_prompt, prompt, search_kwargs)
                stop_time = time.time()
                elapsed_time = round(stop_time - start_time, 2)
                response_content = response["content"]
                # print(response)
                response_metadata = {**response["usage_metadata"],
                                     "model_name": response["response_metadata"]["model_name"],
                                     "total_time": elapsed_time}

            show_chat({
                "role": "assistant",
                "command": command,
                "content": response_content,
                "metadata": response_metadata
            })

        add_messages("assistant", response_content, command, response_metadata)

    st.rerun()


@st.dialog("Local LLM Error")
def ollama_error(msg: str):
    st.markdown(msg)
    st.markdown("""\nYou'll be automatically moved to Cloud mode.""")


@st.dialog("Cloud LLM Error")
def groq_error():
    st.markdown(
        """Add a valid Groq API key to \".streamlit/secrets.toml\", like in the example provide in the 
        \".streamlit/secrets.toml.example\"""")


def force_change_llm(mode: str = "Cloud"):
    """Function to change the LLM mode to Cloud in case of errors in changing to Local mode.
    This can happen when Ollama is not installed on the user PC or if it can't be launched.
    """
    st.session_state.llm_mode = mode
    st.rerun()

def free_ram():
    """Kill the Ollama local model to free up RAM."""

    try:
        ollama.generate(model=st.session_state.rag.llm_model, keep_alive=0)
    except Exception:
        return

def change_llm():
    current_option = st.session_state.get("select", "Cloud")

    if st.session_state.llm_mode != current_option:
        if current_option == "Cloud":
            try:
                with st.spinner("Changing LLM to Cloud..."):
                    free_ram()
                    st.session_state.rag.change_llm(False)
            except Exception:
                groq_error()
                force_change_llm("Local")

        elif current_option == "Local":
            if shutil.which("ollama") is None:
                ollama_error(
                    """For Local LLM mode you need to install **Ollama**. If you want to install Ollama, [click here.]
                    (https://ollama.com/)""")
                force_change_llm()
                return

            try:
                ollama.list()

                with st.spinner("Changing LLM to Local..."):
                    st.session_state.rag.change_llm(True)

                try:
                    st.toast(f"Success: Ollama is already running. Model used: {st.session_state.rag.llm_model}",
                             icon="🎉", duration="infinite")
                except Exception:
                    ollama_error("**Error:** Available RAM is below 5GB and can't load any model.")

            except Exception:
                st.toast("Ollama service is offline. Starting it now...")

                with st.spinner("Starting Ollama..."):
                    subprocess.Popen(["ollama", "serve"],
                                     stdout=subprocess.DEVNULL,
                                     stdin=subprocess.DEVNULL
                                     )

                    time.sleep(2)

                    try:
                        ollama.list()
                        st.session_state.rag.change_llm(True)
                        st.toast(
                            f"Success: Ollama was started successfully. Model used: {st.session_state.rag.llm_model}",
                            icon="🎉")

                    except Exception:
                        ollama_error("**Error:** Tried to start Ollama, but it failed.")

                        force_change_llm()
                        return

        st.session_state.llm_mode = current_option


with st.sidebar:
    st.selectbox(
        label="What type of LLM do you want to use?",
        options=("Cloud", "Local"),
        key="select",
        on_change=change_llm, # Call the function when the selectbox changes
        index=0 if st.session_state.llm_mode == "Cloud" else 1 # Set initial selection
    )

    st.header("AI Model for Analysing Software Requirement")

    st.markdown("Documents from the current session:")

    for index, document in enumerate(st.session_state.documents):
        # print(index, document)
        col1, col2 = st.columns([0.8, 0.2])

        with col1:
            checkbox_key = f"document_checkbox_{index}"
            st.checkbox(
                label=f"**{document['source']}**",
                value=document['isActive'],
                key=checkbox_key,
                on_change=modify_document_state,
                args=[index, checkbox_key]
            )

        with col2:
            st.button(
                label="",
                icon=":material/delete:",
                on_click=delete_document,
                args=[index],
                key=f"btn_{index}"
            )

    with st.container(key="user-guide"):
        if st.button("User Guide", icon=":material/info:"):
            info_modal()

# page styling
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
    
    .response-metadata {{
        font-size: 0.8rem;
        display: flex;
        flex-direction: row-reverse;
        width: 100%;
    }}
        
    .response-metadata > p {{
        font-size: inherit;
    }}
</style>
""")
