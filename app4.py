import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Pealkirjad
st.title("🎓 AI Kursuse Nõustaja - RAGiga")
st.caption("Täisväärtuslik RAG süsteem semantilise otsinguga.")

# Külgriba
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")

# UUS
# Mudeli, andmetabeli ja vektoriseeritud andmete laadimine
# OLULINE: andmed on juba vektoriteks tehtud, loe need .pkl failist
# Eeldame, et puhtad_andmed_embeddings.pkl on pd.dataframe: veergudega (unique_ID, embedding}
# tuleb kasutada streamliti cache_resource, et me mudelit ja andmeid pidevalt uuesti ei laeks 
@st.cache_resource
def get_models():
    # Kasutame SentenceTransformer teeki ja sama mudelit, millega tehti embeddings.npy
    # "BAAI/bge-m3"
    #Todo
    return embedder, df, embeddings_dict

embedder, df, embeddings_dict = get_models()

# 1. Algatame vestluse ajaloo, kui seda veel pole
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Kuvame vestluse senise ajaloo (History)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. KOrjame üles kasutaja sõnumi
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
            # UUS Semantiline otsing (RAG)
            with st.spinner("Otsin sobivaid kursusi..."):
                # Teeme kasutaja küsimusest vektori (query)

                # Ühendame .pkl failis olevad veerud csv-st loetud andmetabeliga
                
                # Arvutame koosinussarnasuse query ja "embedding" veeru vahel. lisame veeru "sarnasus" või "score" andmefreimile
                
                # Sorteerime freimi skoori alusel, Võtame andmetabelist N=5 esimest rida, nimetame seda results_df
                
                # eemaldame vestluse jaoks ebavajalikud veerud: skoor, embedding, unique_ID
                results_df = NotImplemented #todo
                
                context_text = results_df.to_string()

            # 3. LLM vastus koos kontekstiga
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            system_prompt = {
                "role": "system", 
                "content": f"Oled nõustaja. Kasuta järgmisi RAGi leitud kursusi vastamiseks:\n\n{context_text}"
            }
            
            messages_to_send = [system_prompt] + st.session_state.messages
            
            try:
                stream = client.chat.completions.create(
                    model="google/gemma-3-27b-it:free",
                    messages=messages_to_send,
                    stream=True
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Viga: {e}")