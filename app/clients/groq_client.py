import os

from dotenv import load_dotenv
from groq import Groq

_client: Groq | None = None
load_dotenv()

def get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from your .env file!")
        _client = Groq(api_key=api_key)
    return _client

def call_groq_chat(messages: list[dict], model: str = "openai/gpt-oss-20b") -> str:
    """
    Call Groq chat completions and return the assistant's response text.
    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content