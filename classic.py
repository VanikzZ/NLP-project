import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import nlp, embedder, sentiment_analyzer


def classify_sentiment(text):
    result = sentiment_analyzer(text)[0]
    label = result['label']
    score = result['score']
    
    return {label: score}


def classify_classic(text, labels):
    text_emb = embedder.encode([text[:1000]])[0]
    label_embs = embedder.encode(labels)
    sims = cosine_similarity([text_emb], label_embs)[0]
    probs = np.exp(sims) / np.sum(np.exp(sims))
    return dict(zip(labels, probs))


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
    from natasha import Doc, Segmenter, NewsEmbedding, NewsNERTagger
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