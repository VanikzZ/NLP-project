import streamlit as st
import json
import html
import re
from llm_config import LLM_PROVIDERS
from config import CLASSIC_PROVIDERS
from providers import call_llm
from classic import classify_classic, classify_sentiment, summarize_classic, ner_classic
from prompts import CLASSIFICATION_PROMPT, SENTIMENT_PROMPT, SUMMARIZATION_PROMPT, NER_PROMPT
from datasets_loader import show_all_datasets_ui
from rag_agent import create_vectorstore, answer_with_rag
import hashlib

DEFAULT_TEXT = (
    "Совет директоров компании «ТехноИнвест» 15 марта 2026 года объявил о рекордной "
    "квартальной прибыли в размере $5,8 млн. Генеральный директор Алексей Морозов "
    "связал рост с выходом на азиатские рынки. Завод в Шэньчжэне заработает 1 сентября.\n\n"
    "Тем временем сборная Бразилии разгромила Аргентину со счётом 4:1 в финале "
    "чемпионата мира по футболу. Нападающий Винисиус Жуниор оформил хет-трик.\n\n"
    "Учёные из MIT представили новый метод лечения диабета 2 типа. Профессор Джон Смит "
    "утверждает, что клинические испытания начнутся в июле 2027 года. Бюджет проекта — €120 млн."
)


ENTITY_COLORS = {
    "PER": "#2563eb",
    "PERSON": "#2563eb",
    "ORG": "#16a34a",
    "LOC": "#9333ea",
    "LOCATION": "#9333ea",
    "GPE": "#9333ea",
    "DATE": "#ea580c",
    "TIME": "#ea580c",
    "MONEY": "#dc2626",
    "PERCENT": "#dc2626",
    "MISC": "#64748b",
}

ENTITY_NAMES = {
    "PER": "человек",
    "PERSON": "человек",
    "ORG": "организация",
    "LOC": "локация",
    "GPE": "локация",
    "DATE": "дата",
    "TIME": "время",
    "MONEY": "деньги",
    "MISC": "прочее",
}


def extract_ner_items(entities):
    items = []

    if not entities:
        return items

    for ent in entities:
        if isinstance(ent, dict):
            ent_text = ent.get("text") or ent.get("word") or ent.get("entity") or ent.get("entity_text")
            ent_label = ent.get("label") or ent.get("type") or ent.get("entity_group") or ent.get("tag")
        elif isinstance(ent, (list, tuple)) and len(ent) >= 2:
            ent_text, ent_label = ent[0], ent[1]
        else:
            continue

        if ent_text and ent_label:
            items.append((str(ent_text), ent_label))

    return items

def normalize_entity_type(entity_type):
    entity_type = str(entity_type or "MISC").upper().strip()
    if entity_type == "PERSON":
        return "PER"
    if entity_type == "GPE":
        return "LOC"
    return entity_type

def parse_llm_entities(raw_result):
    """Parse LLM NER answer into a list of {text, type} dicts."""
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
        ent_text = item.get("text") or item.get("entity") or item.get("name")
        ent_type = item.get("type") or item.get("label") or item.get("entity_type")
        if ent_text and ent_type:
            entities.append({"text": str(ent_text), "type": normalize_entity_type(ent_type)})
    return entities

def find_entity_spans(text, entities):
    spans = []
    occupied = [False] * len(text)
    for ent in sorted(entities, key=lambda e: len(e.get("text", "")), reverse=True):
        ent_text = str(ent.get("text", "")).strip()
        ent_type = normalize_entity_type(ent.get("type"))
        if not ent_text:
            continue
        start = 0
        while True:
            idx = text.find(ent_text, start)
            if idx == -1:
                break
            end = idx + len(ent_text)
            if not any(occupied[idx:end]):
                spans.append((idx, end, ent_type))
                for i in range(idx, end):
                    occupied[i] = True
            start = end
    return sorted(spans, key=lambda x: x[0])

def entity_badge_html(entity_type):
    ent_type = normalize_entity_type(entity_type)
    color = ENTITY_COLORS.get(ent_type, "#64748b")
    title = ENTITY_NAMES.get(ent_type, ent_type)
    return (
        f'<span title="{html.escape(title)}" '
        f'style="display:inline-block;padding:2px 7px;border-radius:999px;'
        f'background:{color};color:white;font-weight:800;font-size:0.78rem;">'
        f'{html.escape(ent_type)}</span>'
    )

def render_entity_list_item(ent):
    ent_text = html.escape(str(ent.get("text", "")))
    ent_type = normalize_entity_type(ent.get("type"))
    badge = entity_badge_html(ent_type)
    st.markdown(f'• <b>{ent_text}</b> → {badge}', unsafe_allow_html=True)

def render_entity_legend(entities):
    used_types = []
    for ent in entities:
        ent_type = normalize_entity_type(ent.get("type"))
        if ent_type not in used_types:
            used_types.append(ent_type)
    if not used_types:
        return

    badges = []
    for ent_type in used_types:
        color = ENTITY_COLORS.get(ent_type, "#64748b")
        title = ENTITY_NAMES.get(ent_type, ent_type)
        badges.append(
            f'<span style="display:inline-block;margin:0 8px 8px 0;padding:4px 9px;'
            f'border-radius:999px;background:{color};color:white;font-weight:700;">'
            f'{html.escape(ent_type)} · {html.escape(title)}</span>'
        )
    st.markdown("".join(badges), unsafe_allow_html=True)

def render_ner_highlighted_text(text, entities):
    if not entities:
        st.info("Нет сущностей для подсветки.")
        return
    spans = find_entity_spans(text, entities)
    if not spans:
        st.info("Сущности найдены, но их не получилось сопоставить с исходным текстом для подсветки.")
        return

    parts = []
    last = 0
    for start, end, ent_type in spans:
        parts.append(html.escape(text[last:start]))
        color = ENTITY_COLORS.get(ent_type, "#64748b")
        label = html.escape(ent_type)
        value = html.escape(text[start:end])
        parts.append(
            f'<span title="{label}" style="background:{color};color:white;'
            f'padding:2px 5px;border-radius:6px;font-weight:700;white-space:pre-wrap;">'
            f'{value}<sup style="margin-left:4px;font-size:0.7em;">{label}</sup></span>'
        )
        last = end
    parts.append(html.escape(text[last:]))

    html_text = "".join(parts).replace("\n", "<br>")
    st.markdown(
        f'<div style="line-height:2.1;font-size:1.02rem;border:1px solid #334155;'
        f'border-radius:12px;padding:14px;background:rgba(148,163,184,0.08);">{html_text}</div>',
        unsafe_allow_html=True,
    )





def render_sidebar():
    st.sidebar.title("⚙️ Настройки")

    st.sidebar.subheader("📊 Классический NLP")
    classic_method = st.sidebar.selectbox("Метод", list(CLASSIC_PROVIDERS.keys()), index=0)
    st.sidebar.caption(CLASSIC_PROVIDERS[classic_method])

    st.sidebar.divider()
    st.sidebar.subheader("🤖 LLM")
    llm_provider = st.sidebar.selectbox("Провайдер", list(LLM_PROVIDERS.keys()), index=0)
    model = st.sidebar.text_input("Модель", value=LLM_PROVIDERS[llm_provider]["default_model"])

    st.sidebar.divider()
    st.sidebar.subheader("🎛️ Параметры генерации")
    global temperature
    temperature = st.sidebar.slider(
        "Temperature",
        0.0, 1.0, 0.7, 0.05,
        help="0 = точные, повторяемые ответы. 1 = креативные, разнообразные."
    )
    global max_tokens
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        50, 1000, 200, 50,
        help="Максимальная длина генерируемого ответа в токенах."
    )

    return {
        "classic_method": classic_method,
        "llm_provider": llm_provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def render_classification_tab(settings, text):
    st.header("🏷️ Классификация текста")

    mode = st.radio(
        "Режим классификации",
        ["По темам", "Тональность"],
        horizontal=True
    )

    labels = ["экономика", "образование", "спорт", "политика", "наука", "технологии", "медицина"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🔧 Классика")
        if st.button("Запустить классический", key="clf_c"):
            with st.spinner("..."):
                if mode == "Тональность":
                    result = classify_sentiment(text)
                else:
                    result = classify_classic(text, labels)
                for label, prob in sorted(result.items(), key=lambda x: x[1], reverse=True):
                    st.progress(float(prob), text=f"{label}: {prob:.3f}")

    with col2:
        st.subheader(f"🤖 LLM ({settings['llm_provider']})")
        if st.button("Запустить LLM", key="clf_l"):
            with st.spinner("..."):
                if mode == "Тональность":
                    prompt = SENTIMENT_PROMPT.format(text=text)
                else:
                    prompt = CLASSIFICATION_PROMPT.format(labels=", ".join(labels), text=text)
                result = call_llm(
                    provider=settings["llm_provider"],
                    model=settings["model"],
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                st.markdown(f"**Ответ:** {result}")


def render_summarization_tab(settings, text):
    st.header("📝 Суммаризация текста")
    num_sentences = st.slider("Количество предложений", 1, 5, 3)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🔧 Классика")
        if st.button("Запустить классический", key="sum_c"):
            with st.spinner("..."):
                result = summarize_classic(text, num_sentences)
                st.markdown(f"**Результат:**\n\n{result}")

    with col2:
        st.subheader(f"🤖 LLM ({settings['llm_provider']})")
        if st.button("Запустить LLM", key="sum_l"):
            with st.spinner("..."):
                prompt = SUMMARIZATION_PROMPT.format(num_sentences=num_sentences, text=text[:3000])
                result = call_llm(
                    provider=settings["llm_provider"],
                    model=settings["model"],
                    prompt=prompt,
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                )
                st.markdown(f"**Результат:**\n\n{result}")


def render_ner_tab(settings, text):
    st.header("🔍 NER — распознавание сущностей")

    entity_types = st.multiselect(
        "Типы сущностей для LLM",
        ["PER", "ORG", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "MISC"],
        default=["PER", "ORG", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "MISC"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🔧 Классика ({settings['classic_method']})")

        if st.button("Запустить классический", key="ner_c"):
            with st.spinner("Ищем сущности классическим методом..."):
                entities = ner_classic(text, method=settings["classic_method"])

            if entities:
                st.markdown("**Список сущностей:**")
                for ent in entities:
                    render_entity_list_item(ent)

                st.markdown("**Подсветка в исходном тексте:**")
                render_entity_legend(entities)
                render_ner_highlighted_text(text, entities)
            else:
                st.info("Сущности не найдены.")

    with col2:
        st.subheader(f"🤖 LLM ({settings['llm_provider']})")

        if st.button("Запустить LLM", key="ner_l"):
            with st.spinner("Ищем сущности через LLM..."):
                prompt = NER_PROMPT.format(
                    entity_types=", ".join(entity_types),
                    text=text,
                )

                result = call_llm(
                    provider=settings["llm_provider"],
                    model=settings["model"],
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=700,
                )

            st.markdown("**Сырой ответ LLM:**")
            st.code(result, language="json")

            try:
                parsed = parse_llm_entities(result)

                if parsed:
                    st.markdown("**Список сущностей:**")
                    for ent in parsed:
                        render_entity_list_item(ent)

                    st.markdown("**Подсветка в исходном тексте:**")
                    render_entity_legend(parsed)
                    render_ner_highlighted_text(text, parsed)
                else:
                    st.info("LLM вернула пустой список сущностей.")

            except Exception as exc:
                st.error("Не удалось распарсить ответ LLM как JSON.")
                st.caption(str(exc))


def render_rag_tab(settings, text):
    st.header("🤖 RAG — вопрос-ответ по документу")
    st.markdown(
        "Retrieval-Augmented Generation: сначала ищем релевантные чанки в тексте, "
        "потом передаём найденный контекст и вопрос в выбранную LLM."
    )

    if not text.strip():
        st.warning("Введите текст.")
        return

    current_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        st.session_state.num_chunks = 0
        st.session_state.rag_text_hash = None

    if st.session_state.rag_text_hash != current_hash:
        st.session_state.vectorstore = None
        st.session_state.num_chunks = 0
        st.session_state.rag_text_hash = None
        st.info("Текст изменился. Нажмите «Загрузить текст в RAG», чтобы обновить векторную базу.")


    chunk_size = st.slider("Размер чанка (символов)", 50, 1500, 700, 50)
    overlap = st.slider("Перекрытие чанков (символов)", 0, 500, 120, 20)

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("📥 Загрузить текст в RAG"):
            with st.spinner("Разбиваем текст на чанки и строим эмбеддинги..."):
                vectorstore, num_chunks = create_vectorstore(text, chunk_size, overlap)
                st.session_state.vectorstore = vectorstore
                st.session_state.num_chunks = num_chunks
                st.session_state.rag_text_hash = current_hash
                st.success(f"✅ Готово: {num_chunks} чанков")

    with col2:
        if st.session_state.vectorstore is not None and st.session_state.rag_text_hash == current_hash:
            st.caption(f"Векторная база готова: {st.session_state.num_chunks} чанков")
        else:
            st.caption("Векторная база ещё не построена для текущего текста")

    question = st.text_input("Вопрос", placeholder="Например: какой бюджет проекта?")
    top_k = st.slider("Количество чанков для контекста", 1, 5, 3)

    if st.button("💬 Ответить по документу", key="rag_answer"):
        if not question.strip():
            st.warning("Введите вопрос.")
            return

        if st.session_state.vectorstore is None or st.session_state.rag_text_hash != current_hash:
            st.warning("Сначала нажмите «Загрузить текст в RAG» для текущего текста.")
            return

        with st.spinner("Ищем контекст и отправляем его в LLM..."):
            answer, chunks = answer_with_rag(
                st.session_state.vectorstore,
                question,
                provider=settings["llm_provider"],
                model=settings["model"],
                top_k=top_k,
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"]
            )

        st.markdown("**Ответ LLM по найденному контексту:**")
        st.markdown(answer)

        st.markdown("**🔍 Использованные чанки:**")
        for i, chunk in enumerate(chunks, start=1):
            with st.expander(f"Чанк {i} · score={chunk.score:.3f}"):
                st.text(chunk.text)


def create_ui():
    settings = render_sidebar()
    st.title("🧠 NLP Platform — классика vs LLM")
    st.markdown("Сравнение классических методов NLP и больших языковых моделей.")
    text = st.text_area("📄 Введите текст для анализа", height=220, value=DEFAULT_TEXT)
    if not text.strip():
        st.warning("Введите текст для анализа.")
        return
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏷️ Классификация", "📝 Суммаризация", "🔍 NER", "📊 Датасеты", "🤖 RAG (LangChain)"])

    with tab1:
        render_classification_tab(settings, text)
    with tab2:
        render_summarization_tab(settings, text)
    with tab3:
        render_ner_tab(settings, text)
    with tab4:
        show_all_datasets_ui()
    with tab5:
        render_rag_tab(settings, text)
