import os
from typing import Sequence, Any

import ollama
import psutil
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableWithFallbacks
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()


def get_llm(model: str, temperature: float | int = 0.0, max_tokens: int = 3000,
            is_local: bool = False) -> ChatGroq | ChatOllama:
    """Instantiate a LLM based on the user wish.
    If the user wants a Cloud based LLM, it will be used a model from Groq.
    If the user wants a Local LLM, the model will be used through Ollama.

    :param model: The model that will be used for prompts. Check Groq and Ollama models.
    :type model: str
    :param temperature: Sampling temperature. Ranges from 0.0 to 1.0.
    :type temperature: float | int
    :param max_tokens: Number of tokens that can be used by the LLM to generate an answer. Higher than 0.
    :type max_tokens: int
    :param is_local: Determine which LLM provider to use: Groq - Cloud, Ollama - Local.
    :type is_local: bool

    :return: The LLM that will be used for prompts.
    :rtype: ChatGroq | ChatOllama
    """

    if not is_local:
        groq_api_key = st.secrets['GROQ_API_KEY'] or os.environ.get("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")

        llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            streaming=True
        )

        return llm

    else:
        ollama_args = {
            "model": model,
            "temperature": temperature,
            "num_ctx": 8000,
            "num_predict": max_tokens
        }

        try:
            llm = ChatOllama(
                **ollama_args,
                validate_model_on_init=True
            )

        except Exception as e:
            error_msg = str(e).lower()

            if "404" in error_msg or "not found" in error_msg:
                st.toast(f"Model {model} is not installed locally. Starting download...")

                ollama.pull(model)

                llm = ChatOllama(**ollama_args)
            else:
                raise e

        return llm


def create_prompt_llm(is_local: bool = False, max_tokens: int = 3000, temperature: float | int = 0.0) -> \
        RunnableWithFallbacks[PromptValue | str | Sequence[Any], AIMessage] | ChatOllama | None:
    """Function to create the llm used for chatting. It uses 2 model (for Groq - Cloud), in case the first one gets
    an error
    (ex.: you don't have tokens left for that model) it automatically runs the fallback llm or one model depending on
    the users available memory (for Ollama - Local).

    Groq:
        Primary model: llama-3.3-70b-versatile\n
        Fallback model: llama-3.1-8b-instant
    Ollama:
        Memory available:
            - >=24GB: Qwen-3-Coder: 30B
            - >=16GB: Deepseek-Coder-V2: 16B
            - >=8GB: Qwen2.5-Coder: 7B
            - >=5GB: Phi4-Mini - not that great, use just for simple Q&A.

    :param is_local: Determine which LLM provider to use: Groq - Cloud, Ollama - Local.
    :type is_local: bool
    :param temperature: Sampling temperature. Ranges from 0.0 to 1.0.
    :type temperature: float | int
    :param max_tokens: Number of tokens that can be used by the LLM to generate an answer. Higher than 0.
    :type max_tokens: int

    :return: For Cloud: A Cloud based LLM with another Cloud based LLM as fallback, in case the user finishes his
    tokens.
    For Local: A Local Model based on the available RAM memory available.
    :rtype: RunnableWithFallbacks[PromptValue | str | Sequence[Any], AIMessage] | ChatOllama | None
    """

    # check if the parameters can be used for instantiate the models
    if temperature < 0 or temperature > 2:
        raise ValueError("temperature parameter needs to be between 0 and 2")
    if max_tokens <= 0:
        raise ValueError("max tokens parameter cannot be negative")

    if not is_local:
        main_model = "llama-3.3-70b-versatile"
        fallback_model = "llama-3.1-8b-instant"

        main_llm = get_llm(main_model, temperature, max_tokens)
        fallback_llm = get_llm(fallback_model, temperature, max_tokens)

        llm = main_llm.with_fallbacks([fallback_llm])

        return llm

    else:
        # calculate available memory in GB
        memory_available = psutil.virtual_memory().available / (1024 ** 3)

        if memory_available >= 24:
            model = "qwen3-coder:30b"
        elif memory_available >= 16:
            model = "deepseek-coder-v2:16b"
        elif memory_available >= 8:
            model = "qwen2.5-coder:7b"
        elif memory_available >= 5:
            model = "phi4-mini"
        else:
            return None

        llm = get_llm(model, temperature, max_tokens, is_local=True)
        return llm
