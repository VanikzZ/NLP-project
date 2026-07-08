import streamlit as st
import json
from config import LLM_PROVIDERS, CLASSIC_PROVIDERS
from providers import call_llm
from classic import classify_classic, classify_sentiment, summarize_classic, ner_classic
from prompts import CLASSIFICATION_PROMPT, SENTIMENT_PROMT, SUMMARIZATION_PROMPT, NER_PROMPT


DEFAULT_TEXT = (
    "Совет директоров компании «ТехноИнвест» 15 марта 2026 года объявил о рекордной "
    "квартальной прибыли в размере $5,8 млн. Генеральный директор Алексей Морозов "
    "связал рост с выходом на азиатские рынки. Завод в Шэньчжэне заработает 1 сентября.\n\n"
    "Тем временем сборная Бразилии разгромила Аргентину со счётом 4:1 в финале "
    "чемпионата мира по футболу. Нападающий Винисиус Жуниор оформил хет-трик.\n\n"
    "Учёные из MIT представили новый метод лечения диабета 2 типа. Профессор Джон Смит "
    "утверждает, что клинические испытания начнутся в июле 2027 года. Бюджет проекта — €120 млн."
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
                    prompt = SENTIMENT_PROMT.format(text=text)
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
        ["PER", "ORG", "LOC", "DATE", "MONEY"],
        default=["PER", "ORG", "LOC", "DATE", "MONEY"],
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🔧 Классика ({settings['classic_method']})")
        if st.button("Запустить классический", key="ner_c"):
            with st.spinner("..."):
                entities = ner_classic(text, method=settings["classic_method"])
                if entities:
                    for ent in entities:
                        st.markdown(f"- **{ent['text']}** → `{ent['type']}`")
                else:
                    st.info("Сущности не найдены.")

    with col2:
        st.subheader(f"🤖 LLM ({settings['llm_provider']})")
        if st.button("Запустить LLM", key="ner_l"):
            with st.spinner("..."):
                prompt = NER_PROMPT.format(
                    entity_types=", ".join(entity_types),
                    text=text[:2000],
                )
                result = call_llm(
                    provider=settings["llm_provider"],
                    model=settings["model"],
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=500,
                )
                st.markdown(f"**Ответ:**\n\n```\n{result}\n```")
                try:
                    json_str = result.split("```")[1] if "```" in result else result
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    parsed = json.loads(json_str.strip())
                    st.markdown("**Распознанные сущности:**")
                    for ent in parsed:
                        st.markdown(f"- **{ent['text']}** → `{ent['type']}`")
                except Exception:
                    st.caption("Не удалось распарсить JSON.")


def create_ui():
    settings = render_sidebar()
    st.title("🧠 NLP Platform — классика vs LLM")
    st.markdown("Сравнение классических методов NLP и больших языковых моделей.")
    text = st.text_area("📄 Введите текст для анализа", height=220, value=DEFAULT_TEXT)
    if not text.strip():
        st.warning("Введите текст для анализа.")
        return
    tab1, tab2, tab3 = st.tabs(["🏷️ Классификация", "📝 Суммаризация", "🔍 NER"])

    with tab1:
        render_classification_tab(settings, text)
    with tab2:
        render_summarization_tab(settings, text)
    with tab3:
        render_ner_tab(settings, text)