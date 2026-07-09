"""RAG-агент на LangChain."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    template=(
        "Ты — система вопрос-ответ. Отвечай на вопрос, используя ТОЛЬКО контекст ниже.\n"
        "Если в контексте нет ответа, напиши: «В тексте нет информации об этом».\n\n"
        "Контекст: {context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    ),
    input_variables=["context", "question"]
)

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")


def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    vectorstore = Chroma.from_texts(
        texts=chunks, embedding=embeddings, collection_name="user_document"
    )
    return vectorstore, len(chunks)


def create_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )


def run_rag(chain, question):
    result = chain.invoke({"query": question})
    return result["result"], [d.page_content[:150] + "..." for d in result["source_documents"]]