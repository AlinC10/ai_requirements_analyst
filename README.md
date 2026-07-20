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
-   **Loading Indicator:** Add a loading indicator to show that a document is being processed.
-   **Command-Based Actions:** Implement slash commands in the chat to perform specific actions:
    -   `/classify [topic]`: Classify functional vs. non-functional requirements.
    -   `/defects [topic]`: Audit for ambiguities and contradictions.
    -   `/stories [topic]`: Generate Agile User Stories.
    -   `/usecases [topic]`: Generate detailed Use Cases.
    -   `/criteria [topic]`: Generate BDD Acceptance Criteria.
    -   `/diagram [topic]`: Generate a Mermaid diagram.
-   **Token Usage Display:** Show the number of tokens used for each response.
-   **Code Refactoring:** Add comments and refactor `app.py` for better maintainability.
-   **Remove Debug Prints:** Remove all `print()` statements used for debugging.
-   **Database Sync:** Delete documents from the ChromaDB collection when they are removed from the UI.
-   **Conversation History:** Add a feature to save and continue past conversations.
