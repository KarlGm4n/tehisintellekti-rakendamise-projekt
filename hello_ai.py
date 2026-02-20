import json
import pickle
from pathlib import Path
from urllib import request, error

import numpy as np
import pandas as pd
import streamlit as st

# Iluasjad: pealkiri, alapealkiri
st.title("🎓 AI Kursuse Nõustaja")
st.caption("Lihtne vestlusliides automaatvastusega.")

# LLM seadistused (OpenRouter)
api_key = st.sidebar.text_input("OpenRouter API key", type="password")
model_name = st.sidebar.text_input("Model", value="google/gemma-3-27b-it:free")


EMBEDDINGS_FILE = Path("puhtad_andmed_embeddings.pkl")
COURSES_FILE = Path("data/andmed_puhastatud.csv")


@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("BAAI/bge-m3")


@st.cache_data(show_spinner=False)
def load_embeddings_df(file_path):
    with open(file_path, "rb") as handle:
        return pickle.load(handle)


@st.cache_data(show_spinner=False)
def load_courses_df(file_path):
    return pd.read_csv(file_path, low_memory=False)


def prepare_embeddings(embeddings_df):
    if not isinstance(embeddings_df, pd.DataFrame):
        raise ValueError("Embeddings fail peab olema pandas DataFrame.")
    if "unique_ID" not in embeddings_df.columns or "embedding" not in embeddings_df.columns:
        raise ValueError("Embeddings DataFrame peab sisaldama veerge 'unique_ID' ja 'embedding'.")

    ids = embeddings_df["unique_ID"].astype(str).tolist()
    vectors = [np.asarray(v, dtype=np.float32) for v in embeddings_df["embedding"].tolist()]
    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix_norm = matrix / norms
    return ids, matrix_norm


def get_course_key_column(courses_df):
    for col in ("aine_kood", "unique_ID", "id", "course_id"):
        if col in courses_df.columns:
            return col
    return None


def retrieve_top_k(query, embedder, ids, matrix_norm, courses_df, top_k=5):
    query_vec = embedder.encode([query], normalize_embeddings=True)[0]
    scores = matrix_norm @ query_vec
    top_idx = np.argsort(scores)[-top_k:][::-1]
    top_ids = [ids[i] for i in top_idx]
    top_scores = [float(scores[i]) for i in top_idx]

    key_col = get_course_key_column(courses_df)
    if not key_col:
        raise ValueError("Kursuste failist ei leitud sobivat ID veergu.")

    courses_df = courses_df.copy()
    courses_df[key_col] = courses_df[key_col].astype(str)
    indexed = courses_df.set_index(key_col, drop=False)
    results = indexed.reindex(top_ids).reset_index(drop=True)
    results["score"] = top_scores
    return results


def build_context_text(results_df):
    lines = []
    for _, row in results_df.iterrows():
        parts = []
        if "aine_kood" in results_df.columns:
            parts.append(f"Aine kood: {row.get('aine_kood')}")
        if "nimi_et" in results_df.columns:
            parts.append(f"Nimi: {row.get('nimi_et')}")
        if "kirjeldus" in results_df.columns:
            parts.append(f"Kirjeldus: {row.get('kirjeldus')}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


embeddings_df = None
courses_df = None
ids = None
matrix_norm = None
embedder = None

if EMBEDDINGS_FILE.exists():
    try:
        embeddings_df = load_embeddings_df(EMBEDDINGS_FILE)
        ids, matrix_norm = prepare_embeddings(embeddings_df)
        st.sidebar.success("Embeddings laaditud.")
    except Exception as exc:
        st.sidebar.error(f"Embeddings laadimine ebaonnestus: {exc}")
else:
    st.sidebar.warning("Embeddings faili ei leitud.")

if COURSES_FILE.exists():
    try:
        courses_df = load_courses_df(COURSES_FILE)
        st.sidebar.success("Kursused laaditud.")
    except Exception as exc:
        st.sidebar.error(f"Kursuste laadimine ebaonnestus: {exc}")
else:
    st.sidebar.warning("Kursuste faili ei leitud.")


def call_openrouter_chat(api_key_value, model, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key_value}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI Kursuse Noustaja",
    }
    req = request.Request(url, data=data, headers=headers)
    try:
        with request.urlopen(req, timeout=60) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"LLM API error: {exc.code} {exc.reason}. {error_body}")
    except error.URLError as exc:
        raise RuntimeError(f"LLM API connection error: {exc.reason}")

    try:
        return response_data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("LLM API error: unexpected response format.")

# 1. Algatame vestluse ajaloo, kui seda veel pole
if "messages" not in st.session_state:
    st.session_state.messages = []


# 2. Kuvame vestluse senise ajaloo (History)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 3. Korjame üles uue kasutaja sisendi (Action)
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    # Kuvame kohe kasutaja sõnumi ja salvestame selle ka ajalukku
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Kuvame vastuse ja salvestame ajalukku
    if not api_key:
        response = "Lisa OpenRouter API key vasakus menuis, et LLM vastaks."
    elif ids is None or matrix_norm is None or courses_df is None:
        response = "Embeddings voi kursuste fail puudub. Kontrolli faile ja proovi uuesti."
    else:
        with st.spinner("Otsin sobivaid kursusi..."):
            try:
                embedder = get_embedder()
                results_df = retrieve_top_k(prompt, embedder, ids, matrix_norm, courses_df, top_k=5)
            except Exception as exc:
                response = f"Tekkis viga: {exc}"
                results_df = None

        if results_df is not None:
            display_cols = [c for c in ("aine_kood", "nimi_et", "kirjeldus", "score") if c in results_df.columns]
            with st.expander("Leitud kursused"):
                st.dataframe(results_df[display_cols], hide_index=True)

            context_text = build_context_text(results_df)
            system_prompt = {
                "role": "system",
                "content": "Oled noustaja. Kasuta jargnevaid leitud kursusi vastamiseks:\n\n" + context_text,
            }
            messages = [system_prompt] + st.session_state.messages

            with st.spinner("Koostan vastust..."):
                try:
                    response = call_openrouter_chat(api_key, model_name, messages)
                except RuntimeError as exc:
                    response = f"Tekkis viga: {exc}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)