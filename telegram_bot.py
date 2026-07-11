import asyncio
import ast
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
    aliases = {
        "PERSON": "PER",
        "PERSON_NAME": "PER",
        "ЧЕЛОВЕК": "PER",
        "ПЕРСОНА": "PER",
        "NAME": "PER",
        "ORGANIZATION": "ORG",
        "COMPANY": "ORG",
        "КОМПАНИЯ": "ORG",
        "ОРГАНИЗАЦИЯ": "ORG",
        "GPE": "LOC",
        "LOCATION": "LOC",
        "PLACE": "LOC",
        "CITY": "LOC",
        "COUNTRY": "LOC",
        "МЕСТО": "LOC",
        "ЛОКАЦИЯ": "LOC",
        "ГОРОД": "LOC",
        "СТРАНА": "LOC",
        "DATETIME": "DATE",
        "ДАТА": "DATE",
        "MONEY_AMOUNT": "MONEY",
        "AMOUNT": "MONEY",
        "ДЕНЬГИ": "MONEY",
        "СУММА": "MONEY",
    }
    return aliases.get(entity_type, entity_type)


def entity_emoji(entity_type: str) -> str:
    return ENTITY_EMOJIS.get(normalize_entity_type(entity_type), "⬜")


def _append_entity(entities: list[dict[str, str]], seen: set, text, ent_type) -> None:
    if text is None or ent_type is None:
        return

    entity_text = str(text).strip().strip('"').strip("'")
    entity_type = normalize_entity_type(ent_type)
    if not entity_text or entity_type not in {"PER", "ORG", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "MISC"}:
        return

    key = (entity_text.casefold(), entity_type)
    if key not in seen:
        seen.add(key)
        entities.append({"text": entity_text, "type": entity_type})


def parse_llm_entities(raw_result: str) -> list[dict[str, str]]:
    """Разбирает распространённые форматы NER-ответов разных LLM."""
    candidate = str(raw_result or "").strip()
    if not candidate:
        raise ValueError("LLM вернула пустой ответ")

    if candidate.startswith("Ошибка") or candidate.startswith("API ключ"):
        raise RuntimeError(candidate)

    fenced = re.search(r"```(?:json|python)?\s*(.*?)```", candidate, flags=re.S | re.I)
    if fenced:
        candidate = fenced.group(1).strip()

    parsed = None
    parse_errors = []

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(candidate)
            break
        except Exception as exc:
            parse_errors.append(exc)

    if parsed is None:
        # Модель иногда добавляет пояснение до или после JSON.
        json_fragment = re.search(r"(\[.*\]|\{.*\})", candidate, flags=re.S)
        if json_fragment:
            fragment = json_fragment.group(1)
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(fragment)
                    break
                except Exception as exc:
                    parse_errors.append(exc)

    entities: list[dict[str, str]] = []
    seen = set()

    def consume(value, forced_type=None):
        if isinstance(value, dict):
            wrapper = None
            for key in ("entities", "result", "data", "items", "ner", "named_entities", "сущности"):
                if key in value:
                    wrapper = value[key]
                    break
            if wrapper is not None:
                consume(wrapper)
                return

            text_value = (
                value.get("text")
                or value.get("entity")
                or value.get("name")
                or value.get("word")
                or value.get("value")
                or value.get("entity_text")
                or value.get("span")
            )
            type_value = (
                value.get("type")
                or value.get("label")
                or value.get("entity_type")
                or value.get("entity_group")
                or value.get("category")
                or value.get("tag")
                or forced_type
            )
            if text_value and type_value:
                _append_entity(entities, seen, text_value, type_value)
                return

            # Формат {"PER": ["Иван"], "ORG": ["Google"]}
            consumed_typed_map = False
            for key, item in value.items():
                normalized_key = normalize_entity_type(key)
                if normalized_key in {"PER", "ORG", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "MISC"}:
                    consume(item, normalized_key)
                    consumed_typed_map = True
            if consumed_typed_map:
                return

            # Формат {"Иван Петров": "PER"}
            if len(value) == 1:
                only_text, only_type = next(iter(value.items()))
                _append_entity(entities, seen, only_text, only_type)
            return

        if isinstance(value, (list, tuple)):
            if forced_type is not None:
                for item in value:
                    if isinstance(item, str):
                        _append_entity(entities, seen, item, forced_type)
                    else:
                        consume(item, forced_type)
                return

            # Формат ["Иван Петров", "PER"]
            if len(value) >= 2 and isinstance(value[0], str) and isinstance(value[1], str):
                possible_type = normalize_entity_type(value[1])
                if possible_type in {"PER", "ORG", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "MISC"}:
                    _append_entity(entities, seen, value[0], possible_type)
                    return

            for item in value:
                consume(item)
            return

        if isinstance(value, str) and forced_type is not None:
            _append_entity(entities, seen, value, forced_type)

    if parsed is not None:
        consume(parsed)

    # Последний запасной вариант: строки вида "PER: Иван" или "Иван — PER".
    if not entities:
        allowed = r"PER|PERSON|ORG|ORGANIZATION|LOC|LOCATION|GPE|DATE|TIME|MONEY|PERCENT|MISC"
        for line in candidate.splitlines():
            clean = line.strip().lstrip("-•*0123456789. ")
            if not clean:
                continue

            match = re.match(rf"^({allowed})\s*[:=→-]\s*(.+)$", clean, flags=re.I)
            if match:
                _append_entity(entities, seen, match.group(2), match.group(1))
                continue

            match = re.match(rf"^(.+?)\s*(?:→|—|-|:)\s*({allowed})(?:\s*\(.*?\))?$", clean, flags=re.I)
            if match:
                _append_entity(entities, seen, match.group(1), match.group(2))
                continue

            match = re.match(rf"^(.+?)\s*\(({allowed})\)$", clean, flags=re.I)
            if match:
                _append_entity(entities, seen, match.group(1), match.group(2))

    if parsed is None and not entities:
        raise ValueError("В ответе LLM не найден распознаваемый JSON или список сущностей")

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


def make_llm_ner_prompt(text: str, retry: bool = False) -> str:
    retry_note = (
        "Проверь текст повторно особенно внимательно: предыдущая попытка не дала сущностей.\n"
        if retry else ""
    )
    return (
        "Ты выполняешь NER — распознавание именованных сущностей.\n"
        f"{retry_note}"
        "Разрешённые типы: PER — люди; ORG — компании, учреждения и команды; "
        "LOC — города, страны и места; DATE — даты; MONEY — денежные суммы.\n"
        "Найди ВСЕ явно названные сущности. Не пропускай полные имена, названия организаций, "
        "географические названия, даты и суммы.\n"
        "Верни ТОЛЬКО JSON-массив без Markdown и пояснений.\n"
        "Каждый элемент обязан иметь ровно поля text и type.\n"
        "Пример ответа: "
        '[{"text":"Иван Петров","type":"PER"},'
        '{"text":"Google","type":"ORG"},'
        '{"text":"Минск","type":"LOC"},'
        '{"text":"10 июля 2026 года","type":"DATE"},'
        '{"text":"5 миллионов долларов","type":"MONEY"}]\n'
        "Пустой массив [] допустим только если в тексте действительно нет ни одной такой сущности.\n\n"
        f"Текст:\n{text}\n\nJSON:"
    )


def run_llm_ner(provider_name: str, model: str, text: str) -> tuple[list[dict[str, str]], str]:
    """Запрашивает NER и один раз перепроверяет пустой ответ."""
    raw_result = call_llm(
        provider_name,
        model,
        make_llm_ner_prompt(text, retry=False),
        0.0,
        900,
    )
    entities = parse_llm_entities(raw_result)

    if entities:
        return entities, raw_result

    retry_result = call_llm(
        provider_name,
        model,
        make_llm_ner_prompt(text, retry=True),
        0.0,
        900,
    )
    retry_entities = parse_llm_entities(retry_result)
    return retry_entities, retry_result


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
    try:
        entities, _raw_result = await asyncio.to_thread(
            run_llm_ner, provider_name, model, text[:3000]
        )
        await send_long(update, format_entities(entities, "🧠 LLM-сущности:"), reply_markup=main_menu())
    except Exception as exc:
        print(f"LLM NER error: {exc}")
        await send_long(
            update,
            "🧠 LLM-сущности:\nНе удалось обработать ответ модели. Подробность ошибки выведена в терминал.",
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
    from rag_agent_manual import create_vectorstore

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

    from rag_agent_manual import answer_with_rag

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
        try:
            entities, _raw_result = await asyncio.to_thread(
                run_llm_ner, provider_name, model, text[:3000]
            )
            await send_long_to_message(
                query.message,
                format_entities(entities, "🧠 LLM-сущности:"),
                reply_markup=main_menu(),
            )
        except Exception as exc:
            print(f"LLM NER error: {exc}")
            await send_long_to_message(
                query.message,
                "🧠 LLM-сущности:\nНе удалось обработать ответ модели. Подробность ошибки выведена в терминал.",
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
