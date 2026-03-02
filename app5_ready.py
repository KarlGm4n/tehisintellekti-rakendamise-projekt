import json
import re
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "google/gemma-3-27b-it"
MAX_CLARIFICATIONS = 4
RESULTS_N = 5
EUR_PER_INPUT_TOKEN = 0.0000004
EUR_PER_OUTPUT_TOKEN = 0.0000006

FILTER_FIELDS = {
    "semester": {"column": "semester", "type": "categorical"},
    "eap": {"column": "eap", "type": "numeric"},
    "keel": {"column": "keel", "type": "categorical"},
    "linn": {"column": "linn", "type": "categorical"},
    "oppeaste": {"column": "oppeaste", "type": "categorical"},
    "veebiope": {"column": "veebiope", "type": "categorical"},
}

REQUIRED_FILTERS = ["semester", "eap", "keel"]
OPTIONAL_FILTERS = ["linn", "oppeaste", "veebiope"]

USE_SYSTEM_ROLE = not MODEL_NAME.startswith("google/gemma-3")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# --- TAGASISIDE LOGIMISE FUNKTSIOON (sama stiil nagu step6) ---
def log_feedback(timestamp, prompt, filters, context_ids, context_names, response, rating, error_category):
    file_path = 'tagasiside_log.csv'
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Aeg', 'Kasutaja päring', 'Filtrid', 'Leitud ID-d', 'Leitud ained', 'LLM Vastus', 'Hinnang', 'Veatüüp'])
        writer.writerow([timestamp, prompt, filters, str(context_ids), str(context_names), response, rating, error_category])

# pealkiri
st.title("🎓 AI Kursuse Nõustaja")
st.caption("RAG süsteem koos metaandmete filtreerimise ja järelvestlusega.")

@st.cache_resource
def get_models() -> Tuple[SentenceTransformer, pd.DataFrame, pd.DataFrame]:
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv(DATA_DIR / "puhtad_andmed.csv")
    embeddings_df = pd.read_pickle(DATA_DIR / "puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df

embedder, df, embeddings_df = get_models()

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None

def update_usage(usage: Any) -> None:
    if not usage:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    st.session_state.total_tokens += prompt_tokens + completion_tokens
    st.session_state.total_cost += (
        prompt_tokens * EUR_PER_INPUT_TOKEN + completion_tokens * EUR_PER_OUTPUT_TOKEN
    )

def get_allowed_values(source_df: pd.DataFrame) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {}
    for key, meta in FILTER_FIELDS.items():
        if meta["type"] == "categorical":
            col = meta["column"]
            uniques = (
                source_df[col]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            values[key] = uniques
    return values

def apply_filters(source_df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=source_df.index)
    for key, value in filters.items():
        if key not in FILTER_FIELDS:
            continue
        if value in (None, "", "any", "*"):
            continue
        meta = FILTER_FIELDS[key]
        col = meta["column"]
        if meta["type"] == "numeric":
            if isinstance(value, list):
                nums = [float(v) for v in value if v not in (None, "")]
                if nums:
                    mask &= source_df[col].isin(nums)
            else:
                mask &= source_df[col] == float(value)
        else:
            mask &= source_df[col].astype(str).str.lower() == str(value).lower()
    return source_df[mask].copy()

def merge_filters(current: Dict[str, Any], new_values: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(current)
    for key, value in new_values.items():
        if value in (None, "", "any", "*"):
            continue
        merged[key] = value
    return merged

def call_chat(client: OpenAI, messages: List[Dict[str, str]]) -> Tuple[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        stream=False,
    )
    text = response.choices[0].message.content
    return text, response.usage

def build_messages(system_text: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if USE_SYSTEM_ROLE:
        system_msg = {"role": "system", "content": system_text}
    else:
        system_msg = {"role": "user", "content": f"Juhis:\n{system_text}"}
    return [system_msg] + history

def build_filter_system_prompt(
    allowed_values: Dict[str, List[str]],
    current_filters: Dict[str, Any],
    clarifications_left: int,
) -> str:
    return (
        "Oled filtrite kogumise assistent. Eesmärk: saada metaandmete filtrid "
        "kursuste otsinguks. Kohustuslikud filtrid: semester, eap, keel. "
        "Valikulised filtrid: linn, oppeaste, veebiope. "
        "Kui mõni kohustuslik filter on puudu, esita lühike täpsustav küsimus. "
        "Proovi koguda mitu elementi ühe küsimusega. Sul on "
        f"{clarifications_left} küsimust järele. "
        "Kasuta ainult lubatud väärtusi, kui need on teada. "
        "Tagasta ainult JSON järgmise skeemiga: "
        "{\"ready\": bool, \"filters\": {..}, \"next_question\": string}. "
        "Kui oled valmis, siis next_question peab olema tühi. "
        f"Lubatud väärtused: {json.dumps(allowed_values, ensure_ascii=False)}. "
        f"Praegused filtrid: {json.dumps(current_filters, ensure_ascii=False)}."
    )

# külgriba
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.markdown("### Filtrite olek")
    if "filters" in st.session_state and st.session_state.filters:
        st.json(st.session_state.filters)
    else:
        st.caption("Filtreid pole veel määratud.")
    st.markdown("### Kasutusstatistika")
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
    st.metric("Tokenid kokku", st.session_state.total_tokens)
    st.metric("Hinnanguline kulu (€)", f"{st.session_state.total_cost:.5f}")

# seisundi initsialiseerimine
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filter_stage" not in st.session_state:
    st.session_state.filter_stage = "collecting"
if "filters" not in st.session_state:
    st.session_state.filters = {}
if "clarification_count" not in st.session_state:
    st.session_state.clarification_count = 0
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# kuvame ajaloo koos kapotialuse info ja tagasiside vormidega
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Kapotiinfo ja tagasiside ainult assistendi sõnumitele, millel on debug_info
        if message["role"] == "assistant" and "debug_info" in message:
            debug = message["debug_info"]

            # 1. Kapoti all — vahesammude info
            with st.expander("🔍 Vaata kapoti alla (RAG ja filtrid)"):
                st.caption(f"**Aktiivsed filtrid:** {debug.get('filters', 'Info puudub')}")
                st.write(f"**Samm 1 – Metaandmete filtreerimine:** filtrid jätsid andmestikku alles **{debug.get('filtered_count', 0)}** kursust.")

                st.write("**Samm 2 – RAG vektorotsing (Top 5 leitud kursust):**")
                context_df = debug.get('context_df', pd.DataFrame())
                if not context_df.empty:
                    display_cols = ['unique_ID', 'nimi_et', 'eap', 'semester', 'oppeaste', 'score']
                    cols_to_show = [c for c in display_cols if c in context_df.columns]
                    st.dataframe(context_df[cols_to_show], hide_index=True)
                else:
                    st.warning("Ühtegi kursust ei leitud (kas filtrid olid liiga karmid või andmestik tühi).")

                st.write("**Samm 3 – LLM-ile saadetud süsteemiviip:**")
                st.text_area(
                    "Täpne prompt:",
                    debug.get('system_prompt', ''),
                    height=150,
                    disabled=True,
                    key=f"prompt_area_{i}"
                )

            # 2. Tagasiside kogumine
            with st.expander("📝 Hinda vastust (Salvestab logisse)"):
                with st.form(key=f"feedback_form_{i}"):
                    rating = st.radio(
                        "Hinnang vastusele:",
                        ["👍 Hea", "👎 Halb"],
                        horizontal=True,
                        key=f"rating_{i}"
                    )
                    kato = st.selectbox(
                        "Kui vastus oli halb, siis mis läks valesti?",
                        [
                            "",
                            "Filtrid olid liiga karmid/valed (Samm 1)",
                            "Otsing leidis valed ained (Samm 2 – RAG viga)",
                            "LLM hallutsineeris/vastas valesti (Samm 3)"
                        ],
                        key=f"kato_{i}"
                    )
                    if st.form_submit_button("Salvesta hinnang"):
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ctx_df = debug.get('context_df', pd.DataFrame())
                        ctx_ids = ctx_df['unique_ID'].tolist() if not ctx_df.empty else []
                        ctx_names = (
                            ctx_df['nimi_et'].tolist()
                            if (not ctx_df.empty and 'nimi_et' in ctx_df.columns)
                            else []
                        )
                        log_feedback(
                            ts,
                            debug.get('user_prompt', ''),
                            debug.get('filters', ''),
                            ctx_ids,
                            ctx_names,
                            message["content"],
                            rating,
                            kato
                        )
                        st.success("Tagasiside salvestatud tagasiside_log.csv faili!")

# kasutaja päringu töötlemine
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            allowed_values = get_allowed_values(df)

            def run_rag(query_text: str) -> None:
                with st.spinner("Otsin sobivaid kursusi..."):
                    merged_df = pd.merge(df, embeddings_df, on="unique_ID")

                    # --- SAMM 1: Metaandmete filtreerimine ---
                    filtered_df = apply_filters(merged_df, st.session_state.filters)
                    filtered_count = len(filtered_df)

                    if filtered_df.empty:
                        st.warning("Ühtegi kursust ei vasta filtritele.")
                        context_text = "Sobivaid kursusi ei leitud."
                        st.session_state.last_results = None
                        results_df_display = pd.DataFrame()
                    else:
                        # --- SAMM 2: RAG vektorotsing ---
                        query_vec = embedder.encode([query_text])[0]
                        filtered_df["score"] = cosine_similarity(
                            [query_vec], np.stack(filtered_df["embedding"])
                        )[0]
                        results_df = (
                            filtered_df.sort_values("score", ascending=False)
                            .head(RESULTS_N)
                        )
                        results_df_display = results_df.drop(columns=["embedding"], errors="ignore").copy()
                        st.session_state.last_results = results_df_display
                        context_text = results_df.drop(columns=["score", "embedding"], errors="ignore").to_string()

                    filters_summary = json.dumps(st.session_state.filters, ensure_ascii=False)

                    # --- SAMM 3: LLM vastuse genereerimine ---
                    system_text = (
                        "Oled kursusenõustaja. Kasuta ainult antud kursuste infot. "
                        f"Filtrid: {filters_summary}.\n\nKursused:\n{context_text}"
                    )
                    messages_to_send = build_messages(
                        system_text,
                        [m for m in st.session_state.messages if "debug_info" not in m]
                    )

                    try:
                        response_text, usage = call_chat(client, messages_to_send)
                        update_usage(usage)
                        st.markdown(response_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "debug_info": {
                                "user_prompt": query_text,
                                "filters": filters_summary,
                                "filtered_count": filtered_count,
                                "context_df": results_df_display,
                                "system_prompt": system_text,
                            }
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Viga: {e}")

            if st.session_state.filter_stage == "collecting":
                clarifications_left = max(
                    0, MAX_CLARIFICATIONS - st.session_state.clarification_count
                )
                system_text = build_filter_system_prompt(
                    allowed_values, st.session_state.filters, clarifications_left
                )
                history = st.session_state.messages[-8:]
                try:
                    response_text, usage = call_chat(client, build_messages(system_text, history))
                    update_usage(usage)
                    payload = extract_json(response_text) or {}
                    st.session_state.filters = merge_filters(
                        st.session_state.filters, payload.get("filters", {})
                    )

                    ready = bool(payload.get("ready"))
                    next_question = payload.get("next_question", "").strip()

                    if ready or st.session_state.clarification_count >= MAX_CLARIFICATIONS:
                        st.session_state.filter_stage = "ready"
                        run_rag(prompt)
                    else:
                        if not next_question:
                            next_question = "Kas saad täpsustada semestrit, EAP arvu ja keelt?"
                        st.session_state.clarification_count += 1
                        st.markdown(next_question)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": next_question}
                        )
                except Exception as e:
                    st.error(f"Viga: {e}")
            else:
                run_rag(prompt)