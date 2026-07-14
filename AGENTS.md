Project Brief: AI for Software Requirements Analysis

Project Description:
The project involves developing an application that uses Artificial Intelligence to analyze software requirement documents and generate useful design artifacts, such as user stories, acceptance criteria, use cases, and simple diagrams.

The application integrates an LLM into a realistic software engineering workflow to process requirement documents. The primary focus is not on training an AI model from scratch, but on applied AI, specifically Prompt Engineering—testing various prompt formulations for extraction, classification, issue identification, and artifact generation to observe how results vary based on instructions.

Core Application Workflow:

    Upload one or multiple documents describing software requirements (PDF, DOCX, TXT).

    Extract and preprocess text from these documents.

    Identify and categorize functional and non-functional requirements.

    Detect ambiguities, unclarities, or contradictions within the requirements.

    Transform requirements into design artifacts (User Stories, Use Cases, Acceptance Criteria).

    Optionally generate simple diagrams using PlantUML or Mermaid formats.

    Provide a Q&A chat interface allowing the user to ask questions and receive answers based on the analyzed documents.

Tech Stack:

    Language: Python

    Framework: LangChain

    LLM Provider: Groq

        Primary Model: llama-3.3-70b-versatile

        Fallback Model: llama-3.1-8b-instant (used automatically if token/minute limits are reached or tokens are depleted).

    Vector Database: ChromaDB

    Frontend UI: Streamlit

    Configuration: .env file for API keys and global variables.

Data Processing Targets (To be stored in ChromaDB):
The application must be capable of extracting and processing the following elements from documents:

    Narrative Text: Scope, domain, definitions, glossary, constraints (native digital text, e.g., Word).

    Requirement Tables (Crucial structural elements):

        Tables with Requirement ID / Description / Priority / Status.

        Traceability Matrices (Requirement → Test Case → Stakeholder).

        Actors/Roles and Permissions tables.

    Diagrams (Images): UML (use case, sequence, class diagrams), process flowcharts, architecture diagrams (usually inserted as images/screenshots).

    Mockups/Wireframes: UI sketches or screenshots.

    Hierarchical Numbering: e.g., 3.1.2.1, essential for context (child requirements referencing parent requirements).

    Cross-references: e.g., "see section 4.2" or "as per requirement REQ-014".

AI Agent Execution Plan

Provide the following text directly to your AI agent to guide its development process.
Phase 1: Environment Setup & Configuration

    Initialize the Project: Create a new Python virtual environment and set up the project directory.

    Dependencies: Create a requirements.txt installing: streamlit, langchain, langchain-groq, langchain-chroma, python-dotenv, pdfplumber or unstructured (for advanced table/image extraction), and python-docx.

    Environment Variables: Create a .env.example and a .env file containing:

        GROQ_API_KEY

        CHROMA_DB_DIR (local path for vector storage)

        CHUNK_SIZE and CHUNK_OVERLAP settings.

    LLM Fallback Logic: Implement a custom LangChain LLM wrapper or a try-except function for Groq. It must attempt calls using llama-3.3-70b-versatile first. If a RateLimitError or token limit exception occurs, automatically switch to llama-3.1-8b-instant for the request.

Phase 2: Document Ingestion & Advanced Preprocessing

    File Upload Module: Build a Streamlit component accepting .pdf, .docx, and .txt files.

    Smart Text & Structure Extraction: Implement extraction logic capable of parsing complex structures.

        Narrative & Hierarchy: Maintain hierarchical numbering (e.g., 3.1.2.1) in metadata.

        Tables: Use pdfplumber or unstructured to parse tables (Traceability matrices, Requirement ID tables) into Markdown or JSON format before vectorization so they retain their semantic relationships.

        Cross-references: Implement a preprocessing step that resolves or tags explicit cross-references ("see REQ-014").

    Image/Diagram Handling:

        Note to Agent: Since Llama models via Groq are text-only, implement a placeholder function using a basic OCR library (like pytesseract) to extract text from wireframes/UML images found in PDFs, or extract image captions/alt-text to store as metadata.

    Vectorization: Implement a context-aware chunking strategy (using LangChain's RecursiveCharacterTextSplitter) and store the embeddings in local ChromaDB. Include metadata for source file, page number, and section hierarchy.

Phase 3: Core AI Analysis Logic & Prompt Engineering Modules

Develop separate LangChain Chains or Runnables with highly optimized prompts for the following tasks:

    Requirement Classification: Prompt the LLM to scan chunks and tag them as Functional or Non-Functional.

    Defect Detection: Prompt the LLM to analyze requirements for ambiguities, contradictions, or missing constraints (e.g., looking for vague words like "fast", "sometimes", "user-friendly").

    Artifact Generation: Prompt the LLM to take raw functional requirements and output:

        User Stories (As a... I want to... So that...).

        Acceptance Criteria (Given, When, Then format).

        Use Cases (Actors, Preconditions, Main Flow, Alternate Flows).

    Diagram Generation: Prompt the LLM to generate Mermaid.js or PlantUML code blocks representing application flows or Use Cases based on the analyzed text.

Phase 4: Retrieval-Augmented Generation (RAG) & Q&A

    Retriever Setup: Configure ChromaDB as a retriever to fetch relevant document chunks based on user queries.

    Q&A Chain: Build a conversational RAG chain allowing users to ask questions like "What are the security constraints?" or "List all requirements related to the Admin role."

Phase 5: Streamlit User Interface Development

    Layout: Create a multi-tab Streamlit dashboard.

        Tab 1: Upload & Process. File uploaders and a progress spinner for vectorization.

        Tab 2: Analysis Dashboard. Display extracted Functional/Non-Functional requirements and flagged ambiguities in dataframes.

        Tab 3: Artifacts & Diagrams. Display generated User Stories. Render Mermaid.js diagrams directly in Streamlit using streamlit-mermaid or by rendering Markdown components.

        Tab 4: Chatbot. A chat interface utilizing Streamlit's st.chat_message for RAG-based Q&A against the documents.