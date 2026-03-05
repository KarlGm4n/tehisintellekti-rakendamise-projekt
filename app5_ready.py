import json
import re
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from filters_utils import get_allowed_values, apply_filters

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

    st.session_state.total_cost += prompt_tokens * EUR_PER_INPUT_TOKEN + completion_tokens * EUR_PER_OUTPUT_TOKEN

def call_chat(client: OpenAI, messages: List[Dict[str, str]]) -> Tuple[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        stream=False,
    )
    text = response.choices[0].message.content
    return text, response.usage

def merge_filters(current: Dict[str, Any], new_values: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(current)
    for key, value in new_values.items():
        if value in (None, "", "any", "*"):
            continue
        merged[key] = value
    return merged

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
    # Suggest defaults for missing required filters
    defaults = {}
    if not current_filters.get("eap"):
        defaults["eap"] = allowed_values.get("eap", [6])[0] if allowed_values.get("eap") else 6
    if not current_filters.get("semester"):
        defaults["semester"] = allowed_values.get("semester", ["kevad"])[0] if allowed_values.get("semester") else "kevad"
    if not current_filters.get("keel"):
        defaults["keel"] = allowed_values.get("keel", ["ET"])[0] if allowed_values.get("keel") else "ET"
    prompt = (
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
        f"Praegused filtrid: {json.dumps(current_filters, ensure_ascii=False)}. "
        f"Soovitatud vaikimisi: {json.dumps(defaults, ensure_ascii=False)}."
    )
    return prompt

# külgriba
    st.markdown("---")
    st.subheader("🎲 Juhuslik kursuse soovitus")
    if st.button("Soovita juhuslikku kursust", key="random_course_btn"):
        import random
        if not df.empty:
            course = df.sample(1).iloc[0]
            st.success(f"**{course['nimi_et']}** ({course['unique_ID']})")
            st.caption(f"EAP: {course.get('eap', '-')}, Semester: {course.get('semester', '-')}, Keel: {course.get('keel', '-')}")
            st.write(course.get('kirjeldus', 'Kirjeldus puudub'))
        else:
            st.warning("Kursuste andmestik puudub.")
    st.markdown("---")
    st.markdown("### Viimati kasutatud filtrid")
    if "filters" in st.session_state and st.session_state.filters:
        for k, v in st.session_state.filters.items():
            st.caption(f"{k}: {v}")

with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password", key="sidebar_api_key_unique_1")
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

    st.markdown("---")
    st.subheader("🎲 Juhuslik kursuse soovitus")
    if st.button("Soovita juhuslikku kursust", key="random_course_btn"):
        import random
        if not df.empty:
            course = df.sample(1).iloc[0]
            st.success(f"**{course['nimi_et']}** ({course['unique_ID']})")
            st.caption(f"EAP: {course.get('eap', '-')}, Semester: {course.get('semester', '-')}, Keel: {course.get('keel', '-')}")
            st.write(course.get('kirjeldus', 'Kirjeldus puudub'))
        else:
            st.warning("Kursuste andmestik puudub.")
    st.markdown("---")

    st.subheader("Testikomplekt")
    import benchmark_runner
    try:
        benchmark_cases = benchmark_runner.load_benchmark_cases()
        total_cases = len(benchmark_cases)
    except Exception:
        total_cases = 0
    n_cases = st.slider("Mitu testjuhtumit käivitada", min_value=0, max_value=total_cases, value=total_cases, step=1)
    run_bench = st.button("Käivita testikomplekt")
    if run_bench and total_cases > 0:
        st.session_state.benchmark_results = benchmark_runner.run_benchmark(embedder, df, embeddings_df, n_cases)
        st.success(f"Testikomplekt jooksutatud: {n_cases} juhtumit.")

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

# Kuvame benchmarki tulemused, kui need on olemas
if "benchmark_results" in st.session_state and st.session_state.benchmark_results:
    st.subheader("Testikomplekti tulemused")
    results = st.session_state.benchmark_results
    passed = sum(r.passed for r in results)
    st.write(f"\nÕigeid: {passed} / {len(results)}")
    for i, r in enumerate(results, 1):
        with st.expander(f"Juhtum {i}: {'✅' if r.passed else '❌'}"):
            st.write(f"Päring: {r.case.query}")
            st.write(f"Oodatud ID-d: {', '.join(r.case.expected_ids) if r.case.expected_ids else '-'}")
            st.write(f"Leitud ID-d: {', '.join(r.found_ids) if r.found_ids else '-'}")
else:
    # ...existing code for chat history and feedback...
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # ...existing code for debug info and feedback...

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