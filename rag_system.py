from typing import Any

from langchain_core.embeddings import Embeddings

import document_processor as dp
import prompts
from llm import create_prompt_llm
from vector_database import VectorDatabase


class RagSystem:
    """Main classed used in the RAG System that encapsulates the other classes."""

    def __init__(self, embedding_function: Embeddings):
        """
        Initialize the Cloud LLM model used, Chroma Database and the system prompt used by all the prompts.

        :param embedding_function: Embedding function used by the Chroma Vector Database.
        :type embedding_function: Embeddings
        """

        self.llm = create_prompt_llm()
        self.vector_database = VectorDatabase(embedding_function)

        self.system_rules = """
CRITICAL RULES (APPLY TO ALL RESPONSES):

1. **NO HALLUCINATION & SOURCE RELIANCE:** Base your answer ONLY on the `Retrieved Context` provided. Do not invent, 
guess, or assume features, requirements, or defects. If the information is present, ALWAYS cite your sources (e.g., 
[Source: spec.pdf | Page: 4]). 
2. **MISSING CONTEXT PROTOCOL:** If the requested topic is completely missing or unrelated to the context, 
you MUST ONLY reply with a single sentence stating that no information was found, translated into the user's 
language. DO NOT invent features just to fulfill the prompt.
3. **STRICT LANGUAGE MATCHING:** You MUST formulate your ENTIRE response (including table headers, bolded labels, 
code comments, or general text) in the EXACT SAME LANGUAGE as the user's prompt (the Human input). Never mix 
languages. Do NOT start with an English introduction (e.g., "Based on the provided requirements...") if the prompt is 
in another language. Start directly with the requested response.
4. **STRUCTURE & FORMATTING:** Unless a specific technical format (like Mermaid or JSON) is requested, be exhaustive 
and use clear headings, bullet points, and paragraphs. DO NOT provide forced opinions or recommendations unless 
explicitly asked.\n
"""

    def get_response(self, complementary_system_prompt: str, prompt: str, search_kwargs: dict | None = None) -> dict[
        str, Any]:
        """
        Get response for any question and return it to the specified task.

        :param complementary_system_prompt: Specific system prompt that will be concatenated with the system prompt
        from RAGSystem class to formate
    instructions that will achieve better results depending on the user intentions.
        :type complementary_system_prompt: str
        :param prompt: User message that will be sent to the LLM to get a response.
        :type prompt: str
        :param search_kwargs: Filter used in case the user wants to remove any document from the information the LLM
        receives. Default = None
        :type search_kwargs: dict | None
        :return: Response in a JSON format that has the response message and metadata that will be shown to the user.
        :rtype: dict[str, Any]
        """

        # retrieve relevant data about user question from the documents, using search_kwargs as documents filter
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

    def change_llm(self, is_local: bool, max_tokens: int = 3000, temperature: float | int = 0.0) -> None:
        """
        Modify the LLM used by the RAG System based on the given parameters.

        :param is_local: Determine which LLM provider to use: Groq - Cloud, Ollama - Local.
        :type is_local: bool
        :param max_tokens: Number of tokens that can be used by the LLM to generate an answer. Higher than 0. 3k
        tokens by default.
        :type max_tokens: int
        :param temperature: Sampling temperature. Ranges from 0.0 to 1.0. 0 by Default
        :type temperature: float | int
        """

        self.llm = create_prompt_llm(is_local, max_tokens, temperature)

    @property
    def llm_model(self) -> str:
        """
        Property to retrieve the LLM model used by the RAG System. If the LLM is None it will throw an error.

        :return: The name of the LLM model used currently.
        :rtype: str
        """
        return self.llm.model
