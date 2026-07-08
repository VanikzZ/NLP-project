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

from classic import classify_classic, classify_sentiment, summarize_classic, ner_classic

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


def format_scores(scores: dict[str, float]) -> str:
    lines = []
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"{label}: {score:.3f}")
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Привет! Я NLP-бот для мини-проекта.\n\n"
        "Команды:\n"
        "/classify текст — определить тему\n"
        "/sentiment текст — определить тональность\n"
        "/summary текст — сделать краткую выжимку\n"
        "/ner текст — найти сущности\n\n"
        "Можно просто отправить обычный текст — я сделаю классификацию по темам."
    )
    await send_long(update, text)


async def classify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /classify пришли текст. Например: /classify Учёные открыли новый метод лечения диабета")
        return

    await update.message.reply_text("Классифицирую текст...")
    scores = await asyncio.to_thread(classify_classic, text, TOPIC_LABELS)
    await send_long(update, "Результат классификации:\n" + format_scores(scores))


async def sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /sentiment пришли текст. Например: /sentiment Мне очень понравился этот проект")
        return

    await update.message.reply_text("Определяю тональность...")
    result = await asyncio.to_thread(classify_sentiment, text)
    await send_long(update, "Тональность:\n" + format_scores(result))


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /summary пришли текст, который нужно сократить.")
        return

    await update.message.reply_text("Делаю выжимку...")
    result = await asyncio.to_thread(summarize_classic, text, 3)
    await send_long(update, "Выжимка:\n" + result)


async def ner_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_command_text(update)
    if not text:
        await update.message.reply_text("После /ner пришли текст. Например: /ner Алексей Морозов работает в ТехноИнвест в Минске")
        return

    await update.message.reply_text("Ищу сущности...")
    entities = await asyncio.to_thread(ner_classic, text, "spaCy")
    if not entities:
        await update.message.reply_text("Сущности не найдены.")
        return

    lines = [f"{entity['text']} → {entity['type']}" for entity in entities]
    await send_long(update, "Сущности:\n" + "\n".join(lines))


async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    await update.message.reply_text("Понял текст. По умолчанию запускаю классификацию...")
    scores = await asyncio.to_thread(classify_classic, text, TOPIC_LABELS)
    await send_long(update, "Результат классификации:\n" + format_scores(scores))


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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))

    print("Telegram NLP bot is starting...")
    app.run_polling(drop_pending_updates=True, bootstrap_retries=5)


if __name__ == "__main__":
    main()
