from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

import prompts
from document_processor import DocumentProcessor
from llm import create_prompt_llm
from vector_database import VectorDatabase


class RagSystem:
    def __init__(self, embedding_function: Embeddings):
        self.llm = create_prompt_llm()
        self.vector_database = VectorDatabase(embedding_function)

        self.system_rules = """
        CRITICAL RULES:
        1. **Answer only from the Documents:** Base your answer strictly on the `Retrieved Context` provided. If the information is present, cite your sources (e.g., [Source: spec.pdf | Page: 4]).
        2. **STRICT LANGUAGE MATCHING:** You MUST formulate your ENTIRE response in the EXACT SAME LANGUAGE as the user's question (the Human input). Never mix languages.
        3. **NO HALLUCINATION & NO FORCED OPINIONS:** If the exact answer or the topic requested is NOT in the context, you MUST ONLY reply with a single sentence stating that the information was not found in the documents (in the user's language). DO NOT invent answers. DO NOT provide an "Expert Recommendation" unless the user explicitly asks for an opinion or recommendation.
        4. **Structured and Detailed:** If the information is present, be exhaustive. Use clear headings, bullet points, and paragraphs.
"""

    def get_response(self, system_prompt: str, prompt: str) -> AIMessage:
        """Get response for any question and return it to the specified task."""

        # retrieve relevant data about user question from the document
        relevant_chunks = self.vector_database.retrieve_data(prompt)

        # format and join data in a string
        formated_context = DocumentProcessor.format_for_llm(relevant_chunks)

        chat_template = prompts.get_prompt_template(system_prompt, prompt)

        llm_chain = chat_template | self.llm

        response = llm_chain.invoke({'context': formated_context, 'question': prompt})

        return response

    def qa_prompt(self, prompt: str):
        """Get response for a question in QA mode."""

        # add system rules to the system prompt with specific settings for QA mode
        system_prompt = self.system_rules + """
        You are a highly analytical Software Requirements Analyst and Expert Architect. Your primary task is to answer questions based on the provided context.
        
        Retrieved Context:
        {context}     
        """

        response = self.get_response(system_prompt, prompt)

        # print response in console for debugging
        # print(response)
        return response