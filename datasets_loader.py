import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset, Dataset
from collections import Counter
import pandas as pd
import streamlit as st


def load_sentiment_dataset():
    return load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")


def analyze_sentiment(dataset):
    labels = [sample["label"] for sample in dataset]
    counts = Counter(labels)
    return {
        "name": "Rotten Tomatoes",
        "task": "Sentiment Analysis",
        "size": len(dataset),
        "labels": {"Негативный": counts[0], "Позитивный": counts[1]},
        "samples": [
            {"Текст": dataset[i]["text"][:150],
             "Тональность": "Позитивный" if dataset[i]["label"] == 1 else "Негативный"}
            for i in range(5)
        ]
    }


def load_ner_dataset():
    data = {
        "tokens": [
            ["Иван", "Петров", "работает", "в", "Google", "в", "Москве"],
            ["Apple", "анонсировала", "iPhone", "15", "сентября", "2023"],
            ["Мария", "из", "Минска", "получила", "премию"],
            ["Microsoft", "купила", "GitHub", "за", "7.5", "млрд", "долларов"],
            ["Конференция", "пройдёт", "1", "января", "2025", "в", "Лондоне"],
        ],
        "ner_tags": [
            [1, 2, 0, 0, 3, 0, 5],
            [3, 0, 0, 7, 8, 0],
            [1, 0, 5, 0, 0],
            [3, 0, 3, 0, 0, 0, 0],
            [0, 0, 7, 8, 8, 0, 5],
        ]
    }
    return Dataset.from_dict(data)


def analyze_ner(dataset):
    ner_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"]
    all_tags = []
    for sample in dataset:
        all_tags.extend([ner_names[t] for t in sample["ner_tags"] if t > 0])
    counts = Counter(all_tags)

    samples = []
    for i in range(len(dataset)):
        tokens = " ".join(dataset[i]["tokens"])
        entities = [ner_names[t] for t in dataset[i]["ner_tags"] if t > 0]
        samples.append({"Токены": tokens, "Сущности": ", ".join(entities)})

    return {
        "name": "Демо NER",
        "task": "NER",
        "size": len(dataset),
        "labels": dict(counts),
        "samples": samples
    }


def load_summarization_dataset():
    return load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")


def analyze_summarization(dataset):
    article_lens = [len(s["article"].split()) for s in dataset]
    summary_lens = [len(s["highlights"].split()) for s in dataset]

    return {
        "name": "CNN/DailyMail",
        "task": "Summarization",
        "size": len(dataset),
        "avg_article_words": sum(article_lens) // len(article_lens),
        "avg_summary_words": sum(summary_lens) // len(summary_lens),
        "samples": [
            {"Статья (начало)": dataset[i]["article"][:200] + "...",
             "Суммаризация": dataset[i]["highlights"][:150]}
            for i in range(3)
        ]
    }


def get_all_datasets():
    results = {}
    try:
        results["sentiment"] = analyze_sentiment(load_sentiment_dataset())
    except Exception as e:
        results["sentiment"] = {"error": str(e)}
    try:
        results["ner"] = analyze_ner(load_ner_dataset())
    except Exception as e:
        results["ner"] = {"error": str(e)}
    try:
        results["summarization"] = analyze_summarization(load_summarization_dataset())
    except Exception as e:
        results["summarization"] = {"error": str(e)}
    return results


def show_sentiment_ui(data):
    st.subheader(f"📊 {data['name']} — {data['task']}")
    st.markdown(f"**Размер:** {data['size']} примеров")

    dist_df = pd.DataFrame({
        "Метка": list(data["labels"].keys()),
        "Количество": list(data["labels"].values())
    })
    st.markdown("**Распределение меток:**")
    st.bar_chart(dist_df.set_index("Метка"))

    st.markdown("**Примеры:**")
    st.dataframe(pd.DataFrame(data["samples"]), width="stretch")


def show_ner_ui(data):
    st.subheader(f"📊 {data['name']} — {data['task']}")
    st.markdown(f"**Размер:** {data['size']} примеров")

    dist_df = pd.DataFrame({
        "Тип": list(data["labels"].keys()),
        "Количество": list(data["labels"].values())
    })
    st.markdown("**Распределение сущностей:**")
    st.bar_chart(dist_df.set_index("Тип"))

    st.markdown("**Примеры разметки:**")
    st.dataframe(pd.DataFrame(data["samples"]), width="stretch")


def show_summarization_ui(data):
    st.subheader(f"📊 {data['name']} — {data['task']}")
    st.markdown(f"**Размер:** {data['size']} примеров")
    st.markdown(f"**Средняя длина статьи:** {data['avg_article_words']} слов")
    st.markdown(f"**Средняя длина суммаризации:** {data['avg_summary_words']} слов")

    st.markdown("**Примеры:**")
    st.dataframe(pd.DataFrame(data["samples"]), width="stretch")


def show_all_datasets_ui():
    st.header("📊 Работа с датасетами")
    st.markdown("Анализ датасетов для NLP-задач: структура, распределение меток, примеры разметки.")

    with st.spinner("Загружаем датасеты..."):
        results = get_all_datasets()

    tabs = st.tabs(["Тональность", "NER", "Суммаризация"])

    with tabs[0]:
        if "error" in results.get("sentiment", {}):
            st.error(f"Ошибка: {results['sentiment']['error']}")
        else:
            show_sentiment_ui(results["sentiment"])

    with tabs[1]:
        if "error" in results.get("ner", {}):
            st.error(f"Ошибка: {results['ner']['error']}")
        else:
            show_ner_ui(results["ner"])

    with tabs[2]:
        if "error" in results.get("summarization", {}):
            st.error(f"Ошибка: {results['summarization']['error']}")
        else:
            show_summarization_ui(results["summarization"])