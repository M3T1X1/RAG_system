"""Streamlit frontend. Run with: streamlit run app.py"""

import streamlit as st

import backend

st.set_page_config(page_title="RAG — materiały kursowe", page_icon="📚")
st.title("📚 Pytania do materiałów kursowych")

ready = backend.model_available()
if not ready:
    st.warning(f"Model niedostępny. Uruchom Ollamę i `ollama pull {backend.MODEL}`.")

question = st.text_input("Twoje pytanie", placeholder="Np. Czym jest anomalia Belady'ego?")

if st.button("Zapytaj", type="primary", disabled=not question or not ready):
    try:
        with st.spinner("Szukam odpowiedzi…"):
            text, sources = backend.answer(question)
    except NotImplementedError as e:
        st.error(str(e))
    else:
        st.subheader("Odpowiedź")
        st.write(text)
        if sources:
            st.subheader("Źródła")
            for s in sources:
                with st.expander(f"{s.document}, s. {s.page} (podobieństwo {s.score:.2f})"):
                    st.write(s.text)
