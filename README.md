# AI for Software Requirements Analysis

This project is a Streamlit application that uses a Retrieval-Augmented Generation (RAG) system to analyze software requirement documents. It allows users to upload documents, ask questions about them, and generate various design artifacts.

## Core Features

*   **Document Upload:** Supports `.pdf`, `.docx`, and `.txt` files.
*   **AI-Powered Q&A:** Ask questions about your documents and get answers based on their content.
*   **OCR for Images:** Extracts text from images within PDF documents.
*   **On-the-Fly Processing:** Documents are processed in the current session and are not persisted between sessions.

## Tech Stack

*   **Language:** Python
*   **Framework:** LangChain
*   **LLM Provider:** Groq (using `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`)
*   **Vector Database:** ChromaDB
*   **Frontend UI:** Streamlit
*   **Embeddings:** Hugging Face `all-MiniLM-L6-v2`

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Create a `secrets.toml` file:**
    Create a `secrets.toml` file in `.streamlit` and add your Groq API key:
    ```
    GROQ_API_KEY="your-api-key"
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Future Work

This project is under active development. Here are some of the features and improvements planned for the future:

### High Priority
-   **Local LLM:** Add local mode for important documents.
-   **Code Refactoring:** Add comments and refactor `app.py` for better maintainability.
-   **Conversation History:** Add a feature to save and continue past conversations.
