import asyncio
import os
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from llm_config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, get_llm_provider
from providers import call_llm, check_llm
from prompts import CLASSIFICATION_PROMPT, SENTIMENT_PROMPT, SUMMARIZATION_PROMPT, NER_PROMPT

load_dotenv(".env")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")

TOPIC_LABELS = [
    "экономика",
    "образование",
    "спорт",
    "политика",
    "наука",
    "технологии",
    "медицина",
]

ENTITY_TYPES = ["PER", "ORG", "LOC", "DATE", "MONEY"]

# In-memory RAG storage. For a demo/prototype this is enough: each Telegram chat
# can load its own document and ask questions about it while the bot process runs.
RAG_STORES: dict[int, object] = {}
RAG_CHUNK_COUNTS: dict[int, int] = {}


def get_active_llm() -> tuple[str, str]:
    provider_name, cfg = get_llm_provider(DEFAULT_LLM_PROVIDER)
    model = DEFAULT_LLM_MODEL or cfg["default_model"]
    return provider_name, model


async def send_long(update: Update, text: str):
    """Telegram has a message length limit, so split long answers safely."""
    if not update.message:
        return

    max_len = 3900
    for i in range(0, len(text), max_len):
        await update.message.reply_text(text[i:i + max_len])


def get_command_text(update: Update) -> str:
    """Return everything after the command itself."""
    if not update.message or not update.message.text:
        return ""
    parts = update.message.text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def get_command_or_reply_text(update: Update) -> str:
    """Use command argument first, otherwise use text from replied message."""
    text = get_command_text(update)
    if text:
        return text
    if update.message and update.message.reply_to_message and update.message.reply_to_message.text:
        return update.message.reply_to_message.text.strip()
    return ""


def get_chat_id(update: Update) -> int | None:
    return update.effective_chat.id if update.effective_chat else None


def format_scores(scores: dict[str, float]) -> str:
    lines = []
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"{label}: {score:.3f}")
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    provider_name, model = get_active_llm()
    text = (
        "Привет! Я NLP-бот для мини-проекта.\n\n"
        "Классические команды:\n"
        "/classify текст — определить тему классическим способом\n"
        "/sentiment текст — определить тональность классическим способом\n"
        "/summary текст — сделать классическую выжимку\n"
        "/ner текст — найти сущности классическим способом\n\n"
        "LLM-команды:\n"
        "/llm_test — проверить подключение к LLM API\n"
        "/llm_classify текст — определить тему через LLM\n"
        "/llm_sentiment текст — определить тональность через LLM\n"
        "/llm_summary текст — сделать выжимку через LLM\n"
        "/llm_ner текст — найти сущности через LLM\n"
        "/compare текст — сравнить классическую и LLM-классификацию\n\n"
        "RAG-команды:\n"
        "/rag_load текст — загрузить документ/текст в RAG\n"
        "/rag_ask вопрос — ответить по загруженному тексту через LLM\n"
        "/rag_clear — очистить RAG-память текущего чата\n\n"
        f"Активная LLM: {provider_name} / {model}\n\n"
        "Можно просто отправить обычный текст — я сделаю классическую классификацию."
    )
    await send_long(update, text)


async def classify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /classify пришли текст. Например: /classify Учёные открыли новый метод лечения диабета")
        return

    await update.message.reply_text("Классифицирую текст классическим способом...")
    from classic import classify_classic
    scores = await asyncio.to_thread(classify_classic, text, TOPIC_LABELS)
    await send_long(update, "Классическая классификация:\n" + format_scores(scores))


async def sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /sentiment пришли текст. Например: /sentiment Мне очень понравился этот проект")
        return

    await update.message.reply_text("Определяю тональность классическим способом...")
    from classic import classify_sentiment
    result = await asyncio.to_thread(classify_sentiment, text)
    await send_long(update, "Классическая тональность:\n" + format_scores(result))


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /summary пришли текст, который нужно сократить.")
        return

    await update.message.reply_text("Делаю классическую выжимку...")
    from classic import summarize_classic
    result = await asyncio.to_thread(summarize_classic, text, 3)
    await send_long(update, "Классическая выжимка:\n" + result)


async def ner_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /ner пришли текст. Например: /ner Алексей Морозов работает в ТехноИнвест в Минске")
        return

    await update.message.reply_text("Ищу сущности классическим способом...")
    from classic import ner_classic
    entities = await asyncio.to_thread(ner_classic, text, "spaCy")
    if not entities:
        await update.message.reply_text("Сущности не найдены.")
        return

    lines = [f"{entity['text']} → {entity['type']}" for entity in entities]
    await send_long(update, "Классические сущности:\n" + "\n".join(lines))


async def llm_test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Проверяю подключение к LLM API...")
    provider_name, model = get_active_llm()
    result = await asyncio.to_thread(check_llm, provider_name, model)
    await send_long(update, result)


async def llm_classify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /llm_classify пришли текст.")
        return

    await update.message.reply_text("Классифицирую через LLM...")
    provider_name, model = get_active_llm()
    prompt = CLASSIFICATION_PROMPT.format(labels=", ".join(TOPIC_LABELS), text=text[:3000])
    result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 80)
    await send_long(update, "LLM-классификация:\n" + result)


async def llm_sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /llm_sentiment пришли текст.")
        return

    await update.message.reply_text("Определяю тональность через LLM...")
    provider_name, model = get_active_llm()
    prompt = SENTIMENT_PROMPT.format(text=text[:3000])
    result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 80)
    await send_long(update, "LLM-тональность:\n" + result)


async def llm_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /llm_summary пришли текст.")
        return

    await update.message.reply_text("Делаю выжимку через LLM...")
    provider_name, model = get_active_llm()
    prompt = SUMMARIZATION_PROMPT.format(num_sentences=3, text=text[:4000])
    result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.2, 400)
    await send_long(update, "LLM-выжимка:\n" + result)


async def llm_ner_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /llm_ner пришли текст.")
        return

    await update.message.reply_text("Ищу сущности через LLM...")
    provider_name, model = get_active_llm()
    prompt = NER_PROMPT.format(entity_types=", ".join(ENTITY_TYPES), text=text[:3000])
    result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 500)
    await send_long(update, "LLM-сущности:\n" + result)


async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /compare пришли текст.")
        return

    await update.message.reply_text("Сравниваю классификацию: классика vs LLM...")

    from classic import classify_classic
    classic_scores = await asyncio.to_thread(classify_classic, text, TOPIC_LABELS)

    provider_name, model = get_active_llm()
    prompt = CLASSIFICATION_PROMPT.format(labels=", ".join(TOPIC_LABELS), text=text[:3000])
    llm_result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 80)

    answer = (
        "Классическая классификация:\n"
        f"{format_scores(classic_scores)}\n\n"
        "LLM-классификация:\n"
        f"{llm_result}"
    )
    await send_long(update, answer)


async def rag_load_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_or_reply_text(update)
    chat_id = get_chat_id(update)

    if chat_id is None:
        return
    if not text:
        await update.message.reply_text(
            "После /rag_load пришли текст документа. Ещё можно ответить командой /rag_load на сообщение с длинным текстом."
        )
        return

    await update.message.reply_text("Загружаю текст в RAG: режу на чанки и строю эмбеддинги...")
    from rag_agent import create_vectorstore

    vectorstore, num_chunks = await asyncio.to_thread(create_vectorstore, text)
    if num_chunks == 0:
        await update.message.reply_text("Не получилось создать чанки: текст пустой или слишком короткий.")
        return

    RAG_STORES[chat_id] = vectorstore
    RAG_CHUNK_COUNTS[chat_id] = num_chunks
    await update.message.reply_text(
        f"RAG готов: текст разбит на {num_chunks} чанков. Теперь задай вопрос командой /rag_ask."
    )


async def rag_ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = get_command_text(update)
    chat_id = get_chat_id(update)

    if chat_id is None:
        return
    if not question:
        await update.message.reply_text("После /rag_ask пришли вопрос по загруженному тексту.")
        return
    if chat_id not in RAG_STORES:
        await update.message.reply_text("Сначала загрузи текст командой /rag_load текст.")
        return

    await update.message.reply_text("Ищу релевантные чанки и отправляю их в LLM...")

    from rag_agent import answer_with_rag

    provider_name, model = get_active_llm()
    answer, chunks = await asyncio.to_thread(
        answer_with_rag,
        RAG_STORES[chat_id],
        question,
        provider_name,
        model,
        3,
        700,
    )

    sources = "\n".join(
        f"{i}. Чанк {chunk.index + 1}, score={chunk.score:.3f}: {chunk.text[:220]}..."
        for i, chunk in enumerate(chunks, start=1)
    )

    result = "RAG-ответ по документу:\n" + answer
    if sources:
        result += "\n\nИспользованные чанки:\n" + sources

    await send_long(update, result)


async def rag_clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = get_chat_id(update)
    if chat_id is None:
        return
    RAG_STORES.pop(chat_id, None)
    RAG_CHUNK_COUNTS.pop(chat_id, None)
    await update.message.reply_text("RAG-память текущего чата очищена.")


async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    await update.message.reply_text("Понял текст. По умолчанию запускаю классическую классификацию...")
    from classic import classify_classic
    scores = await asyncio.to_thread(classify_classic, text, TOPIC_LABELS)
    await send_long(update, "Классическая классификация:\n" + format_scores(scores))


def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN или BOT_TOKEN не найден. Проверь файл .env")

    request = HTTPXRequest(
        connect_timeout=30,
        read_timeout=30,
        write_timeout=30,
        pool_timeout=30,
    )

    app = ApplicationBuilder().token(TOKEN).request(request).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("classify", classify_command))
    app.add_handler(CommandHandler("sentiment", sentiment_command))
    app.add_handler(CommandHandler("summary", summary_command))
    app.add_handler(CommandHandler("ner", ner_command))
    app.add_handler(CommandHandler("llm_test", llm_test_command))
    app.add_handler(CommandHandler("llm_classify", llm_classify_command))
    app.add_handler(CommandHandler("llm_sentiment", llm_sentiment_command))
    app.add_handler(CommandHandler("llm_summary", llm_summary_command))
    app.add_handler(CommandHandler("llm_ner", llm_ner_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("rag_load", rag_load_command))
    app.add_handler(CommandHandler("rag_ask", rag_ask_command))
    app.add_handler(CommandHandler("rag_clear", rag_clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))

    print("Telegram NLP bot is starting...")
    app.run_polling(drop_pending_updates=True, bootstrap_retries=5)


if __name__ == "__main__":
    main()
