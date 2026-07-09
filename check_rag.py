from llm_config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, get_llm_provider
from rag_agent import create_vectorstore, answer_with_rag

TEXT = """
Компания ТехноИнвест объявила о рекордной прибыли в размере 5,8 млн долларов.
После этого компания открыла офис в Минске и запустила проект по анализу медицинских данных.
Врачи разработали новый метод лечения диабета и планируют клинические испытания.
""".strip()

QUESTION = "Что говорится про лечение диабета?"

provider_name, cfg = get_llm_provider(DEFAULT_LLM_PROVIDER)
model = DEFAULT_LLM_MODEL or cfg["default_model"]

vectorstore, num_chunks = create_vectorstore(TEXT)
print(f"RAG chunks: {num_chunks}")
print(f"Provider: {provider_name}")
print(f"Model: {model}")
print(f"Question: {QUESTION}")

answer, chunks = answer_with_rag(vectorstore, QUESTION, provider_name, model)
print("\nAnswer:")
print(answer)
print("\nSources:")
for chunk in chunks:
    print(f"- chunk={chunk.index + 1}, score={chunk.score:.3f}, text={chunk.text[:180]}")
