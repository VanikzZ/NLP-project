import os
import requests

from llm_config import LLM_PROVIDERS, get_llm_provider


def _headers(provider_name: str, api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # OpenRouter accepts OpenAI-compatible requests. These two headers are optional,
    # but they help identify the app in OpenRouter dashboards/leaderboards.
    if provider_name.startswith("OpenRouter"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE", "NLP mini-project bot")

    return headers


def call_llm(provider: str | None, model: str | None, prompt: str, temperature: float = 0.3, max_tokens: int = 300) -> str:
    provider_name, cfg = get_llm_provider(provider)

    if not cfg.get("key"):
        return (
            f"API ключ для {provider_name} не задан. "
            "Добавь MISTRAL_API_KEY или OPENROUTER_API_KEY в файл .env."
        )

    model_name = model or cfg["default_model"]

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            cfg["url"],
            headers=_headers(provider_name, cfg["key"]),
            json=payload,
            timeout=60,
        )
    except requests.exceptions.Timeout:
        return "Ошибка: LLM API не ответил за 60 секунд. Проверь интернет, VPN/прокси или попробуй позже."
    except requests.exceptions.RequestException as e:
        return f"Ошибка сети при обращении к LLM API: {e}"

    if response.status_code != 200:
        return f"Ошибка API {response.status_code}: {response.text[:700]}"

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Ошибка разбора ответа LLM: {e}. Ответ API: {response.text[:700]}"


def check_llm(provider: str | None = None, model: str | None = None) -> str:
    provider_name, cfg = get_llm_provider(provider)
    model_name = model or cfg["default_model"]
    result = call_llm(
        provider=provider_name,
        model=model_name,
        prompt="Ответь ровно одним словом: OK",
        temperature=0.0,
        max_tokens=20,
    )
    return f"Provider: {provider_name}\nModel: {model_name}\nAnswer: {result}"
