import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from llm_config import LLM_PROVIDERS

print("Loading models...")

nlp = spacy.load("ru_core_news_lg")
embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")
sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

print("Models are loaded!")


CLASSIC_PROVIDERS = {
    "spaCy": "",
    "Natasha": "Только NER",
}
