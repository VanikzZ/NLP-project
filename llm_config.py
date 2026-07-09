import os
from dotenv import load_dotenv

load_dotenv(".env")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Defaults are kept in one place so Streamlit, Telegram and tests use the same config.
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "Mistral")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "")

LLM_PROVIDERS = {
    "Mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "key": MISTRAL_API_KEY,
        "default_model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        "missing_key_hint": "Добавь MISTRAL_API_KEY в файл .env.",
    },
    "OpenRouter Qwen": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key": OPENROUTER_API_KEY,
        "default_model": os.getenv(
            "OPENROUTER_MODEL",
            "qwen/qwen3-next-80b-a3b-instruct:free",
        ),
        "missing_key_hint": "Добавь OPENROUTER_API_KEY в файл .env.",
    },
    "OpenAI": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key": OPENAI_API_KEY,
        "default_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "missing_key_hint": "Добавь OPENAI_API_KEY в файл .env. ChatGPT Plus не заменяет API-ключ.",
    },
}


def get_llm_provider(provider_name: str | None = None) -> tuple[str, dict]:
    """Return a valid provider name and its config."""
    name = provider_name or DEFAULT_LLM_PROVIDER
    if name not in LLM_PROVIDERS:
        name = next(iter(LLM_PROVIDERS))
    return name, LLM_PROVIDERS[name]
