import os
from dotenv import load_dotenv
import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

print("Loading models...")

nlp = spacy.load("ru_core_news_lg")
embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")
sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

print("Models are loaded!")


LLM_PROVIDERS = {
    "Mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "key": MISTRAL_API_KEY,
        "default_model": "mistral-small",
    },
    "OpenRouter Tencent": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key": OPENROUTER_API_KEY,
        "default_model": "tencent/hy3:free",
    },
    "OpenRouter Qwen": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key": OPENROUTER_API_KEY,
        "default_model": "qwen/qwen3-next-80b-a3b-instruct:free"
    }
}

CLASSIC_PROVIDERS = {
    "spaCy": "",
    "Natasha": "Только NER",
}