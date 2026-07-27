from typing import Any

from langchain_core.embeddings import Embeddings

import document_processor as dp
import prompts
from llm import create_prompt_llm
from vector_database import VectorDatabase


class RagSystem:
    def __init__(self, embedding_function: Embeddings):
        self.llm = create_prompt_llm()
        self.vector_database = VectorDatabase(embedding_function)

        self.system_rules = """
        CRITICAL RULES:
        1. **Answer only from the Documents:** Base your answer strictly on the `Retrieved Context` provided. If the 
        information is present, cite your sources (e.g., [Source: spec.pdf | Page: 4] or [Source: spec.pdf | Chapter: 
        Cars] or etc).
        2. **STRICT LANGUAGE MATCHING:** You MUST formulate your ENTIRE response in the EXACT SAME LANGUAGE as the 
        user's question (the Human input). Never mix languages.
        3. **NO HALLUCINATION & NO FORCED OPINIONS:** If the exact answer or the topic requested is NOT in the 
        context, you MUST ONLY reply with a single sentence stating that the information was not found in the 
        documents (in the user's language). DO NOT invent answers. DO NOT provide an "Expert Recommendation" unless 
        the user explicitly asks for an opinion or recommendation.
        4. **Structured and Detailed:** If the information is present, be exhaustive. Use clear headings, 
        bullet points, and paragraphs.\n\n
"""

    def get_response(self, complementary_system_prompt: str, prompt: str, search_kwargs: dict | None = None) -> dict[
        str, Any]:
        """Get response for any question and return it to the specified task."""

        # retrieve relevant data about user question from the document
        relevant_chunks = self.vector_database.retrieve_data(prompt, search_kwargs)

        # format and join data in a string
        formated_context = dp.format_for_llm(relevant_chunks)
        if not formated_context.strip():
            formated_context = "No relevant context was retrieved from the uploaded documents."

        chat_template = prompts.get_prompt_template(self.system_rules + complementary_system_prompt)

        llm_chain = chat_template | self.llm

        response = llm_chain.invoke({'context': formated_context, 'question': prompt})
        response_dump = response.model_dump()

        return response_dump
