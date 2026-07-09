"""RAG-агент на LangChain + ChromaDB."""

from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from prompts import RAG_QA_PROMPT
from providers import call_llm

EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"

embeddings = HuggingFaceEmbeddings(
    model_name="distiluse-base-multilingual-cased-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    index: int


def create_vectorstore(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="user_document"
    )
    return vectorstore, len(chunks)


def answer_with_rag(vectorstore, question, provider=None, model=None, top_k=3, max_tokens=300, temperature=0.0):
    # Используем similarity_search_with_score вместо invoke
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)
    
    if not docs_with_scores:
        return "Чанки не найдены.", []

    docs = [doc for doc, _ in docs_with_scores]
    scores = [score for _, score in docs_with_scores]

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = RAG_QA_PROMPT.format(context=context, question=question)
    answer = call_llm(provider, model, prompt, temperature=temperature, max_tokens=max_tokens)

    chunks = [
        RetrievedChunk(text=docs[i].page_content, score=float(scores[i]), index=i)
        for i in range(len(docs))
    ]
    return answer, chunks