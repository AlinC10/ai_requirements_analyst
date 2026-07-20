import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from streamlit import secrets

load_dotenv()


def get_llm(model: str, temperature: float | int = 0.0, max_tokens: int = 3000) -> ChatGroq:
    """Instantiate a llm from Groq platform."""
    # groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_api_key = secrets['GROQ_API_KEY']

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


def create_prompt_llm(max_tokens: int = 3000, temperature: float | int = 0.0):
    """Function to create the llm used for chatting. It uses 2 model, in case the first one gets an error
    (ex.: you don't have tokens left for that model) it automatically runs the fallback llm.
    Primary model: llama-3.3-70b-versatile
    Fallback model: llama-3.1-8b-instant
    """

    # check if the parameters can be used for instantiate the models
    if temperature < 0 and temperature > 2:
        raise ValueError("temperature parameter needs to be between 0 and 2")
    if max_tokens <= 0:
        raise ValueError("max tokens parameter cannot be negative")

    main_model = "llama-3.3-70b-versatile"
    fallback_model = "llama-3.1-8b-instant"

    main_llm = get_llm(main_model, temperature, max_tokens)
    fallback_llm = get_llm(fallback_model, temperature, max_tokens)

    llm = main_llm.with_fallbacks([fallback_llm])

    return llm
