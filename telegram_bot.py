import asyncio
import json
import os
import re
from dotenv import load_dotenv

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from llm_config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, get_llm_provider
from providers import call_llm, check_llm
from prompts import CLASSIFICATION_PROMPT, SENTIMENT_PROMPT, NER_PROMPT

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

ENTITY_EMOJIS = {
    "PER": "🟦",
    "PERSON": "🟦",
    "ORG": "🟩",
    "LOC": "🟪",
    "GPE": "🟪",
    "DATE": "🟧",
    "TIME": "🟧",
    "MONEY": "🟥",
    "PERCENT": "🟥",
    "MISC": "⬜",
}

ENTITY_NAMES = {
    "PER": "человек",
    "PERSON": "человек",
    "ORG": "организация",
    "LOC": "место",
    "GPE": "место",
    "DATE": "дата",
    "TIME": "время",
    "MONEY": "деньги",
    "PERCENT": "процент",
    "MISC": "прочее",
}

# Для демо этого достаточно: у каждого чата своя RAG-память, пока работает процесс бота.
RAG_STORES: dict[int, object] = {}
RAG_CHUNK_COUNTS: dict[int, int] = {}
RAG_TEXTS: dict[int, str] = {}


def get_active_llm() -> tuple[str, str]:
    provider_name, cfg = get_llm_provider(DEFAULT_LLM_PROVIDER)
    model = DEFAULT_LLM_MODEL or cfg["default_model"]
    return provider_name, model


def get_chat_id(update: Update) -> int | None:
    return update.effective_chat.id if update.effective_chat else None


async def send_long_to_message(message, text: str, reply_markup=None):
    """Telegram ограничивает длину сообщения, поэтому длинные ответы режем на части."""
    if not message:
        return

    max_len = 3900
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or [""]
    for i, chunk in enumerate(chunks):
        await message.reply_text(chunk, reply_markup=reply_markup if i == len(chunks) - 1 else None)


async def send_long(update: Update, text: str, reply_markup=None):
    await send_long_to_message(update.message, text, reply_markup=reply_markup)


def get_command_text(update: Update) -> str:
    if not update.message or not update.message.text:
        return ""
    parts = update.message.text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def get_command_or_reply_text(update: Update) -> str:
    text = get_command_text(update)
    if text:
        return text
    if update.message and update.message.reply_to_message and update.message.reply_to_message.text:
        return update.message.reply_to_message.text.strip()
    return ""


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🏷️ Классификация", callback_data="classic_classify"),
            InlineKeyboardButton("🔍 NER классика", callback_data="classic_ner"),
        ],
        [
            InlineKeyboardButton("🤖 LLM классификация", callback_data="llm_classify"),
            InlineKeyboardButton("🧠 LLM NER", callback_data="llm_ner"),
        ],
        [
            InlineKeyboardButton("📝 LLM-выжимка", callback_data="llm_summary"),
        ],
        [
            InlineKeyboardButton("📥 Загрузить в RAG", callback_data="rag_load_last"),
            InlineKeyboardButton("❓ Спросить RAG", callback_data="rag_ask_wait"),
        ],
        [
            InlineKeyboardButton("📄 Показать RAG-текст", callback_data="rag_show_text"),
            InlineKeyboardButton("🧹 Очистить RAG", callback_data="rag_clear_memory"),
        ],
    ])


def format_scores(scores: dict[str, float]) -> str:
    lines = []
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"• {label}: {score:.3f}")
    return "\n".join(lines)


def normalize_entity_type(entity_type) -> str:
    entity_type = str(entity_type or "MISC").upper().strip()
    if entity_type == "PERSON":
        return "PER"
    if entity_type == "GPE":
        return "LOC"
    return entity_type


def entity_emoji(entity_type: str) -> str:
    return ENTITY_EMOJIS.get(normalize_entity_type(entity_type), "⬜")


def parse_llm_entities(raw_result: str) -> list[dict[str, str]]:
    """LLM должна вернуть JSON. Тут убираем code fence и приводим к одному формату."""
    candidate = raw_result.strip()

    if "```" in candidate:
        match = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.S | re.I)
        if match:
            candidate = match.group(1).strip()

    parsed = json.loads(candidate)
    if isinstance(parsed, dict):
        parsed = parsed.get("entities", [])

    entities = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("entity") or item.get("name")
        ent_type = item.get("type") or item.get("label") or item.get("entity_type")
        if text and ent_type:
            entities.append({"text": str(text), "type": normalize_entity_type(ent_type)})
    return entities


def format_entities(entities: list[dict[str, str]], title: str) -> str:
    if not entities:
        return f"{title}:\nСущности не найдены."

    lines = [title]
    for ent in entities:
        ent_text = ent.get("text", "")
        ent_type = normalize_entity_type(ent.get("type"))
        ent_name = ENTITY_NAMES.get(ent_type, ent_type)
        lines.append(f"{entity_emoji(ent_type)} {ent_text} → {ent_type} ({ent_name})")

    return "\n".join(lines)


def make_llm_summary_prompt(text: str, num_sentences: int = 2) -> str:
    return (
        f"Сделай краткую выжимку текста на русском языке в {num_sentences} предложениях.\n"
        "Не повторяй текст полностью. Оставь только главное: кто, что сделал, где, когда, ключевые цифры.\n"
        "Если текст короткий, всё равно переформулируй его короче.\n\n"
        f"Текст:\n{text}\n\nВыжимка:"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    provider_name, model = get_active_llm()
    text = (
        "Привет! Я NLP-бот для мини-проекта.\n\n"
        "Как пользоваться удобнее всего:\n"
        "1. Просто отправь текст обычным сообщением.\n"
        "2. Я сохраню его и покажу кнопки действий.\n"
        "3. Нажимай нужную кнопку: классификация, NER, выжимка или RAG.\n\n"
        "Команды тоже остались:\n"
        "/classify текст — классическая классификация\n"
        "/ner текст — классический NER\n"
        "/llm_summary текст — LLM-выжимка\n"
        "/llm_ner текст — NER через LLM\n"
        "/rag_load текст — загрузить текст в RAG\n"
        "/rag_ask вопрос — задать вопрос по RAG\n"
        "/rag_show — показать текущий RAG-текст\n"
        "/rag_clear — очистить RAG\n\n"
        f"Активная LLM: {provider_name} / {model}"
    )
    await send_long(update, text, reply_markup=main_menu())


async def classify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /classify отправь текст.")
        return

    context.user_data["last_text"] = text
    await update.message.reply_text("Классифицирую текст классическим способом...")
    from classic import classify_classic
    scores = await asyncio.to_thread(classify_classic, text, TOPIC_LABELS)
    await send_long(update, "🏷️ Классическая классификация:\n" + format_scores(scores), reply_markup=main_menu())


async def sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /sentiment отправь текст.")
        return

    context.user_data["last_text"] = text
    await update.message.reply_text("Определяю тональность классическим способом...")
    from classic import classify_sentiment
    result = await asyncio.to_thread(classify_sentiment, text)
    await send_long(update, "🙂 Классическая тональность:\n" + format_scores(result), reply_markup=main_menu())


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /summary отправь текст.")
        return

    context.user_data["last_text"] = text
    await update.message.reply_text("Делаю классическую выжимку...")
    from classic import summarize_classic
    result = await asyncio.to_thread(summarize_classic, text, 2)
    await send_long(update, "📝 Классическая выжимка:\n" + result, reply_markup=main_menu())


async def ner_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /ner отправь текст.")
        return

    context.user_data["last_text"] = text
    await update.message.reply_text("Ищу сущности классическим способом...")
    from classic import ner_classic
    entities = await asyncio.to_thread(ner_classic, text, "spaCy")
    await send_long(update, format_entities(entities, "🔍 Классические сущности:"), reply_markup=main_menu())


async def llm_test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Проверяю подключение к LLM API...")
    provider_name, model = get_active_llm()
    result = await asyncio.to_thread(check_llm, provider_name, model)
    await send_long(update, result)


async def llm_classify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /llm_classify отправь текст.")
        return

    context.user_data["last_text"] = text
    await update.message.reply_text("Классифицирую через LLM...")
    provider_name, model = get_active_llm()
    prompt = CLASSIFICATION_PROMPT.format(labels=", ".join(TOPIC_LABELS), text=text[:3000])
    result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 80)
    await send_long(update, "🤖 LLM-классификация:\n" + result, reply_markup=main_menu())


async def llm_sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /llm_sentiment отправь текст.")
        return

    context.user_data["last_text"] = text
    await update.message.reply_text("Определяю тональность через LLM...")
    provider_name, model = get_active_llm()
    prompt = SENTIMENT_PROMPT.format(text=text[:3000])
    result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 80)
    await send_long(update, "🤖 LLM-тональность:\n" + result, reply_markup=main_menu())


async def llm_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /llm_summary отправь текст.")
        return

    context.user_data["last_text"] = text
    await update.message.reply_text("Делаю LLM-выжимку...")
    provider_name, model = get_active_llm()
    prompt = make_llm_summary_prompt(text[:4000], num_sentences=2)
    result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.2, 350)
    await send_long(update, "📝 LLM-выжимка:\n" + result, reply_markup=main_menu())


async def llm_ner_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /llm_ner отправь текст.")
        return

    context.user_data["last_text"] = text
    await update.message.reply_text("Ищу сущности через LLM...")
    provider_name, model = get_active_llm()
    prompt = NER_PROMPT.format(entity_types=", ".join(ENTITY_TYPES), text=text[:3000])
    result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 700)

    try:
        entities = parse_llm_entities(result)
        await send_long(update, format_entities(entities, "🧠 LLM-сущности:"), reply_markup=main_menu())
    except Exception:
        await send_long(
            update,
            "🧠 LLM-сущности:\nНе удалось красиво разобрать ответ модели. Попробуй повторить запрос или сделать текст короче.",
            reply_markup=main_menu(),
        )


async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /compare отправь текст.")
        return

    context.user_data["last_text"] = text
    await update.message.reply_text("Сравниваю классификацию: классика vs LLM...")

    from classic import classify_classic
    classic_scores = await asyncio.to_thread(classify_classic, text, TOPIC_LABELS)

    provider_name, model = get_active_llm()
    prompt = CLASSIFICATION_PROMPT.format(labels=", ".join(TOPIC_LABELS), text=text[:3000])
    llm_result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 80)

    answer = (
        "🏷️ Классическая классификация:\n"
        f"{format_scores(classic_scores)}\n\n"
        "🤖 LLM-классификация:\n"
        f"{llm_result}"
    )
    await send_long(update, answer, reply_markup=main_menu())


async def rag_load_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    chat_id = get_chat_id(update)
    if chat_id is None:
        return

    await update.effective_message.reply_text("📥 Загружаю текст в RAG: режу на чанки и строю эмбеддинги...")
    from rag_agent import create_vectorstore

    vectorstore, num_chunks = await asyncio.to_thread(create_vectorstore, text)
    if num_chunks == 0:
        await update.effective_message.reply_text("Не получилось создать чанки: текст пустой или слишком короткий.")
        return

    RAG_STORES[chat_id] = vectorstore
    RAG_CHUNK_COUNTS[chat_id] = num_chunks
    RAG_TEXTS[chat_id] = text
    await update.effective_message.reply_text(
        f"✅ RAG готов: текст разбит на {num_chunks} чанков.\n"
        "Теперь нажми «❓ Спросить RAG» или отправь /rag_ask вопрос.",
        reply_markup=main_menu(),
    )


async def rag_load_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_or_reply_text(update)
    if not text:
        await update.message.reply_text("После /rag_load отправь текст документа или ответь командой /rag_load на сообщение с текстом.")
        return
    context.user_data["last_text"] = text
    await rag_load_text(update, context, text)


async def answer_rag_question(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str):
    chat_id = get_chat_id(update)
    if chat_id is None:
        return

    if chat_id not in RAG_STORES:
        await update.effective_message.reply_text("Сначала загрузи текст в RAG: отправь текст и нажми «📥 Загрузить в RAG».")
        return

    await update.effective_message.reply_text("🔎 Ищу релевантные чанки и отправляю их в LLM...")

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

    result = "💬 RAG-ответ по документу:\n" + answer
    if sources:
        result += "\n\n🔍 Использованные чанки:\n" + sources

    await send_long_to_message(update.effective_message, result, reply_markup=main_menu())


async def rag_ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = get_command_text(update)
    if not question:
        context.user_data["awaiting_rag_question"] = True
        await update.message.reply_text("❓ Введи вопрос по загруженному в RAG тексту следующим сообщением.")
        return
    await answer_rag_question(update, context, question)




async def rag_show_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = get_chat_id(update)
    if chat_id is None:
        return

    if chat_id not in RAG_TEXTS:
        await update.effective_message.reply_text(
            "📭 Сейчас в RAG нет загруженного текста. Сначала отправь текст и нажми «📥 Загрузить в RAG».",
            reply_markup=main_menu(),
        )
        return

    text = RAG_TEXTS[chat_id]
    chunks = RAG_CHUNK_COUNTS.get(chat_id, 0)

    answer = (
        f"📄 Текущий текст в RAG\\n"
        f"Чанков: {chunks}\\n\\n"
        f"{text}"
    )

    await send_long_to_message(update.effective_message, answer, reply_markup=main_menu())

async def rag_clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = get_chat_id(update)
    if chat_id is None:
        return
    RAG_STORES.pop(chat_id, None)
    RAG_CHUNK_COUNTS.pop(chat_id, None)
    RAG_TEXTS.pop(chat_id, None)
    context.user_data.pop("awaiting_rag_question", None)
    await update.message.reply_text("🧹 RAG-память текущего чата очищена.")


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not query.message:
        return

    await query.answer()
    action = query.data
    text = context.user_data.get("last_text", "")


    if action == "rag_show_text":
        if get_chat_id(update) not in RAG_TEXTS:
            await query.message.reply_text(
                "📭 Сейчас в RAG нет загруженного текста. Сначала отправь текст и нажми «📥 Загрузить в RAG».",
                reply_markup=main_menu(),
            )
            return

        chat_id = get_chat_id(update)
        rag_text = RAG_TEXTS[chat_id]
        chunks = RAG_CHUNK_COUNTS.get(chat_id, 0)

        await send_long_to_message(
            query.message,
            f"📄 Текущий текст в RAG\nЧанков: {chunks}\n\n{rag_text}",
            reply_markup=main_menu(),
        )
        return

    if action == "rag_clear_memory":
        chat_id = get_chat_id(update)
        if chat_id is not None:
            RAG_STORES.pop(chat_id, None)
            RAG_CHUNK_COUNTS.pop(chat_id, None)
            RAG_TEXTS.pop(chat_id, None)
        context.user_data.pop("awaiting_rag_question", None)

        await query.message.reply_text("🧹 RAG-память текущего чата очищена.", reply_markup=main_menu())
        return

    if action == "rag_ask_wait":
        if get_chat_id(update) not in RAG_STORES:
            await query.message.reply_text("Сначала загрузи текст в RAG: отправь текст и нажми «📥 Загрузить в RAG».", reply_markup=main_menu())
            return
        context.user_data["awaiting_rag_question"] = True
        await query.message.reply_text("❓ Теперь введи вопрос по загруженному тексту обычным сообщением.")
        return

    if not text:
        await query.message.reply_text("Сначала отправь текст обычным сообщением, потом выбери действие кнопкой.", reply_markup=main_menu())
        return

    fake_update = update

    if action == "classic_classify":
        await query.message.reply_text("Классифицирую текст классическим способом...")
        from classic import classify_classic
        scores = await asyncio.to_thread(classify_classic, text, TOPIC_LABELS)
        await send_long_to_message(query.message, "🏷️ Классическая классификация:\n" + format_scores(scores), reply_markup=main_menu())

    elif action == "classic_ner":
        await query.message.reply_text("Ищу сущности классическим способом...")
        from classic import ner_classic
        entities = await asyncio.to_thread(ner_classic, text, "spaCy")
        await send_long_to_message(query.message, format_entities(entities, "🔍 Классические сущности:"), reply_markup=main_menu())

    elif action == "llm_classify":
        await query.message.reply_text("Классифицирую через LLM...")
        provider_name, model = get_active_llm()
        prompt = CLASSIFICATION_PROMPT.format(labels=", ".join(TOPIC_LABELS), text=text[:3000])
        result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 80)
        await send_long_to_message(query.message, "🤖 LLM-классификация:\n" + result, reply_markup=main_menu())

    elif action == "llm_ner":
        await query.message.reply_text("Ищу сущности через LLM...")
        provider_name, model = get_active_llm()
        prompt = NER_PROMPT.format(entity_types=", ".join(ENTITY_TYPES), text=text[:3000])
        result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.0, 700)
        try:
            entities = parse_llm_entities(result)
            await send_long_to_message(query.message, format_entities(entities, "🧠 LLM-сущности:"), reply_markup=main_menu())
        except Exception:
            await send_long_to_message(
                query.message,
                "🧠 LLM-сущности:\nНе удалось красиво разобрать ответ модели. Попробуй повторить запрос или сделать текст короче.",
                reply_markup=main_menu(),
            )

    elif action == "llm_summary":
        await query.message.reply_text("Делаю LLM-выжимку...")
        provider_name, model = get_active_llm()
        prompt = make_llm_summary_prompt(text[:4000], num_sentences=2)
        result = await asyncio.to_thread(call_llm, provider_name, model, prompt, 0.2, 350)
        await send_long_to_message(query.message, "📝 LLM-выжимка:\n" + result, reply_markup=main_menu())

    elif action == "rag_load_last":
        await rag_load_text(fake_update, context, text)


async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()

    if context.user_data.get("awaiting_rag_question"):
        context.user_data["awaiting_rag_question"] = False
        await answer_rag_question(update, context, text)
        return

    context.user_data["last_text"] = text
    await update.message.reply_text(
        "✅ Текст сохранён. Теперь выбери действие кнопкой ниже.",
        reply_markup=main_menu(),
    )


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
    app.add_handler(CommandHandler("rag_show", rag_show_command))
    app.add_handler(CallbackQueryHandler(menu_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))

    print("Telegram NLP bot is starting...")
    app.run_polling(drop_pending_updates=True, bootstrap_retries=5)


if __name__ == "__main__":
    main()
