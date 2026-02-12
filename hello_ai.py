import streamlit as st

st.set_page_config(page_title="Minu Esimene Äpp", page_icon="🤖")

st.title("Tere, tehisintellekti rakendaja! 👋")
st.write("Kui sa näed seda teksti, siis sinu töökeskkond on 100% korras.")

# Lihtne interaktiivsus
name = st.text_input("Kirjuta siia oma nimi:")
if name:
    st.success(f"Väga meeldiv, {name}! Sinu arvuti on kursuseks valmis.")