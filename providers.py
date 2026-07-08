"""Вызов LLM API."""

import requests
from config import LLM_PROVIDERS


def call_llm(provider, model, prompt, temperature=0.3, max_tokens=300):
    cfg = LLM_PROVIDERS.get(provider)
    if not cfg:
        return f"Провайдер '{provider}' не найден."
    if not cfg["key"]:
        return f"API ключ для {provider} не задан"

    model_name = model or cfg["default_model"]

    try:
        response = requests.post(
            cfg["url"],
            headers={
                "Authorization": f"Bearer {cfg['key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Ошибка API {response.status_code}: {response.text[:300]}"
    except Exception as e:
        return f"Ошибка: {str(e)}"