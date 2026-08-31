# AI for Software Requirements Analysis

This project is a Streamlit application that uses a Retrieval-Augmented Generation (RAG) system to analyze software requirement documents. It allows users to upload documents, ask questions about them, and generate various design artifacts.

## Core Features

*   **Document Upload:** Supports `.pdf`, `.docx`, and `.txt` files.
*   **Dual LLM Modes:**
    *   **Cloud Mode:** Utilizes Groq's powerful `openai/gpt-oss-120b` for high-performance analysis, with an automatic fallback to `openai/gpt-oss-20b`.
    *   **Local Mode:** Run analysis entirely on your local machine using Ollama for enhanced privacy and offline capability.
*   **AI-Powered Q&A:** Ask questions about your documents and get answers based on their content.
*   **OCR for Images:** Extracts text from images within PDF documents.
*   **On-the-Fly Processing:** Documents are processed in the current session and are not persisted between sessions.

## Tech Stack

*   **Language:** Python
*   **Framework:** LangChain
*   **LLM Providers:**
    *   **Cloud:** Groq (using `openai/gpt-oss-120b` and `openai/gpt-oss-20b`)
    *   **Local:** Ollama
*   **Vector Database:** ChromaDB
*   **Frontend UI:** Streamlit
*   **Embeddings:** Hugging Face `all-MiniLM-L6-v2`

## How to Run the Project

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Set Up the Environment

**a. Create a virtual environment and install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**b. (Optional) Install Ollama for Local LLM Mode:**

If you wish to use the local LLM mode, you only need to download and run Ollama.

1.  **Download Ollama:** Visit the [Ollama website](https://ollama.com/) and follow the installation instructions for your operating system.
2.  **Run Ollama:** Make sure the Ollama service is running in the background.

The application will automatically detect your available system RAM and select an appropriate model. If the required model is not already installed on your machine, the application will attempt to download it for you the first time you switch to Local Mode.

> **Note for Advanced Users:** The model selection logic is defined in `llm.py`. You can modify this file to change which models are used based on your preferences and hardware.

### 3. Configure API Keys (for Cloud Mode)

If you plan to use the cloud mode, create a `secrets.toml` file in the `.streamlit` directory and add your Groq API key:

```toml
GROQ_API_KEY="your-api-key"
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

## Future Work

This project is under active development. Here are some of the features and improvements planned for the future:

### High Priority
-   **Code Refactoring:** Add comments and refactor `app.py` for better maintainability.
-   **Conversation History:** Add a feature to save and continue past conversations.
-   **Improved Error Handling:** Enhance error messages and provide more guidance to the user.
