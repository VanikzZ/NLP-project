import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import nlp, embedder, sentiment_analyzer
from natasha import Doc, Segmenter, NewsEmbedding, NewsNERTagger


TOPIC_DESCRIPTIONS = {
    "экономика": (
        "Тема: экономика, бизнес, финансы, компании, прибыль, рынок, "
        "инвестиции, деньги, бюджет, акции, производство, торговля, продажи."
    ),
    "образование": (
        "Тема: образование, школа, университет, студенты, ученики, экзамены, "
        "обучение, курсы, преподаватели, лекции, учебные программы."
    ),
    "спорт": (
        "Тема: спорт, футбол, баскетбол, матч, турнир, чемпионат, команда, "
        "игрок, тренер, гол, победа, поражение, финал, соревнования."
    ),
    "политика": (
        "Тема: политика, правительство, президент, парламент, выборы, закон, "
        "партия, министр, государство, дипломатия, санкции, власть."
    ),
    "наука": (
        "Тема: наука, исследование, учёные, эксперимент, открытие, теория, "
        "лаборатория, университет, статья, гипотеза, клинические испытания."
    ),
    "технологии": (
        "Тема: технологии, IT, программирование, искусственный интеллект, "
        "компьютеры, робототехника, приложение, софт, алгоритмы, данные, стартап."
    ),
    "медицина": (
        "Тема: медицина, здоровье, болезнь, лечение, врач, пациент, диагноз, "
        "лекарство, терапия, клиника, диабет, симптомы, медицинские испытания."
    ),
}


def classify_sentiment(text):
    result = sentiment_analyzer(text)[0]
    label = result["label"]
    score = result["score"]

    return {label: score}


def _softmax(scores, temperature=0.08):
    """Convert similarity scores into sharper pseudo-probabilities."""
    scores = np.array(scores, dtype=float)
    scores = scores / temperature
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()


def _get_label_text(label):
    """Turn a short label into a richer text that the embedder can compare better."""
    normalized = label.strip().lower()
    return TOPIC_DESCRIPTIONS.get(
        normalized,
        f"Тема: {label}. Текст относится к теме {label}, содержит связанные события, факты и термины.",
    )


def _split_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences or [text.strip()]


def classify_classic(text, labels):
    clean_labels = [label.strip() for label in labels if label.strip()]
    if not text.strip() or not clean_labels:
        return {}

    label_texts = [_get_label_text(label) for label in clean_labels]
    label_embs = embedder.encode(label_texts, normalize_embeddings=True)

    sentences = _split_into_sentences(text)
    sentence_embs = embedder.encode(sentences, normalize_embeddings=True)

    sim_matrix = cosine_similarity(sentence_embs, label_embs)

    # A long text may contain several themes. One average embedding for the whole text
    # mixes them together, so we score each topic by its best matching sentence.
    scores = sim_matrix.max(axis=0)

    probs = _softmax(scores, temperature=0.08)
    return dict(zip(clean_labels, probs))


def summarize_classic(text, num_sentences=2):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    if not sentences or len(sentences) <= num_sentences:
        return " ".join(sentences)

    embs = embedder.encode(sentences)
    text_emb = np.mean(embs, axis=0).reshape(1, -1)
    sims = cosine_similarity(embs, text_emb).flatten()
    top_indices = np.argsort(sims)[::-1][:num_sentences]
    top_sentences = [sentences[i] for i in sorted(top_indices)]
    return " ".join(top_sentences)


def ner_classic(text, method="spaCy"):
    if method == "Natasha":
        return _ner_natasha(text)
    else:
        return _ner_spacy(text)


def _ner_spacy(text):
    doc = nlp(text)
    entities = []
    seen = set()
    for ent in doc.ents:
        if ent.text.strip() and ent.text not in seen:
            entities.append({"text": ent.text, "type": ent.label_})
            seen.add(ent.text)
    return entities


def _ner_natasha(text):
    segmenter = Segmenter()
    emb = NewsEmbedding()
    ner_tagger = NewsNERTagger(emb)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    entities = []
    seen = set()
    for span in doc.spans:
        if span.text.strip() and span.text not in seen:
            entities.append({"text": span.text, "type": span.type})
            seen.add(span.text)
    return entities
