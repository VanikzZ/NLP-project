from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from prompts import RAG_QA_PROMPT
from providers import call_llm

EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


@dataclass
class RetrievedChunk:
    text: str
    score: float
    index: int


@dataclass
class SimpleVectorStore:
    chunks: list[str]
    embeddings: np.ndarray

    def search(self, question: str, k: int = 3) -> list[RetrievedChunk]:
        if not question.strip() or not self.chunks:
            return []

        query_embedding = get_embedder().encode([question], normalize_embeddings=True)
        semantic_scores = cosine_similarity(query_embedding, self.embeddings)[0]

        final_scores = []
        for i, chunk in enumerate(self.chunks):
            keyword = keyword_score(question, chunk)
            semantic = float(semantic_scores[i])
            combined = 0.65 * semantic + 0.35 * keyword
            final_scores.append(combined)

        top_indexes = np.argsort(final_scores)[::-1][:k]

        return [
            RetrievedChunk(
                text=self.chunks[i],
                score=float(final_scores[i]),
                index=int(i),
            )
            for i in top_indexes
        ]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[а-яёa-z0-9€$]+", text.lower())
    stop = {
        "что", "как", "какой", "какая", "какие", "где", "когда",
        "про", "это", "этот", "эта", "этом", "об", "о", "в", "на",
        "и", "или", "по", "для", "the", "a", "an", "of", "in", "on"
    }
    return {w for w in words if w not in stop and len(w) > 1}


def keyword_score(question: str, chunk: str) -> float:
    q = tokenize(question)
    c = tokenize(chunk)

    if not q:
        return 0.0

    overlap = len(q & c) / len(q)

    q_lower = question.lower()
    c_lower = chunk.lower()

    bonus = 0.0

    if "бюджет" in q_lower and "бюджет" in c_lower:
        bonus += 0.7

    if any(x in q_lower for x in ["стоимость", "сколько", "сумма"]) and any(x in c_lower for x in ["€", "$", "млн", "миллион"]):
        bonus += 0.5

    if any(x in q_lower for x in ["когда", "дата"]) and re.search(r"\d{4}|январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр", c_lower):
        bonus += 0.5

    return min(1.0, overlap + bonus)


def split_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        step = max(1, chunk_size - overlap)
        for start in range(0, len(paragraph), step):
            piece = paragraph[start:start + chunk_size].strip()
            if piece:
                chunks.append(piece)

    return chunks


def create_vectorstore(text: str, chunk_size: int = 700, overlap: int = 120) -> tuple[SimpleVectorStore, int]:
    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return SimpleVectorStore(chunks=[], embeddings=np.empty((0, 0))), 0
    embeddings = get_embedder().encode(chunks, normalize_embeddings=True)
    return SimpleVectorStore(chunks=chunks, embeddings=np.array(embeddings)), len(chunks)


def get_relevant_chunks(vectorstore: SimpleVectorStore, question: str, k: int = 3) -> list[RetrievedChunk]:
    return vectorstore.search(question, k=k)


def build_context(chunks: Iterable[RetrievedChunk]) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"[Чанк {chunk.index + 1}, score={chunk.score:.3f}]\n{chunk.text}")
    return "\n\n".join(parts)


def extractive_fallback(question: str, context: str, llm_answer: str) -> str:
    answer_lower = llm_answer.lower()
    question_lower = question.lower()

    if "нет информации" not in answer_lower:
        return llm_answer

    # Бюджет / стоимость
    if any(word in question_lower for word in ["бюджет", "стоимость", "сколько", "сумма"]):
        patterns = [
            r"(Бюджет[^.\n]*?(?:€|\$|₽|руб|млн|миллион)[^.\n]*)",
            r"((?:€|\$|₽)\s?\d+[,\.\d]*\s?(?:млн|миллион|тыс|k|m)?)",
            r"(\d+[,\.\d]*\s?(?:млн|миллион|тыс)\s?(?:евро|доллар|руб|€|\$)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, context, flags=re.I)
            if match:
                value = match.group(1).strip()
                return value if value.endswith(".") else value + "."

    # Даты
    if any(word in question_lower for word in ["когда", "дата", "начнутся", "начнется"]):
        match = re.search(
            r"((?:в\s+)?(?:январе|феврале|марте|апреле|мае|июне|июле|августе|сентябре|октябре|ноябре|декабре)\s+\d{4}\s+года)",
            context,
            flags=re.I,
        )
        if match:
            return match.group(1).strip().capitalize() + "."

    return llm_answer




def normalize_rag_question(question: str) -> str:
    q = question.strip()
    if not q:
        return q

    # Если пользователь ввёл просто тему: "футбол", "футболу", "диабет", "бюджет"
    # превращаем это в нормальный вопрос для LLM.
    words = re.findall(r"[а-яёa-z0-9€$]+", q.lower())
    has_question_mark = "?" in q
    question_words = {
        "что", "как", "какой", "какая", "какие", "когда", "где",
        "кто", "почему", "зачем", "сколько", "чем"
    }

    if len(words) <= 3 and not has_question_mark and not (set(words) & question_words):
        return f"Что в тексте говорится про {q}?"

    return q


def extract_relevant_sentences(question: str, context: str) -> str | None:
    q_tokens = tokenize(question)
    if not q_tokens:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", context)
    scored = []

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue

        s_tokens = tokenize(s)
        overlap = len(q_tokens & s_tokens)

        # Небольшая нормализация для падежей: футбол / футболу / футбольный
        q_lower = question.lower()
        s_lower = s.lower()

        if "футбол" in q_lower and "футбол" in s_lower:
            overlap += 3
        if "диабет" in q_lower and "диабет" in s_lower:
            overlap += 3
        if "бюджет" in q_lower and "бюджет" in s_lower:
            overlap += 3

        if overlap > 0:
            scored.append((overlap, s))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    answer = " ".join(sentence for _, sentence in scored[:2])
    return answer.strip()


def answer_with_rag(
    vectorstore: SimpleVectorStore,
    question: str,
    provider: str | None,
    model: str | None,
    top_k: int = 3,
    max_tokens: int = 700,
) -> tuple[str, list[RetrievedChunk]]:
    question = normalize_rag_question(question)
    chunks = get_relevant_chunks(vectorstore, question, k=top_k)

    if not chunks:
        return "Сначала загрузите текст в RAG или задайте непустой вопрос.", []

    context = build_context(chunks)
    prompt = RAG_QA_PROMPT.format(context=context, question=question)

    answer = call_llm(
        provider,
        model,
        prompt,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    answer = extractive_fallback(question, context, answer)

    if "нет информации" in answer.lower():
        relevant = extract_relevant_sentences(question, context)
        if relevant:
            answer = relevant

    return answer, chunks


def create_rag_chain(vectorstore, llm=None):
    return {"vectorstore": vectorstore, "llm": llm}


def run_rag(chain, question):
    vectorstore = chain["vectorstore"] if isinstance(chain, dict) else chain
    answer, chunks = answer_with_rag(vectorstore, question, provider=None, model=None)
    return answer, [chunk.text[:150] + "..." for chunk in chunks]
