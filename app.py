"""Streamlit frontend. Run with: streamlit run app.py"""

import streamlit as st

import backend

st.set_page_config(page_title="RAG — materiały kursowe")
st.title("Pytania do materiałów kursowych")


def liczba_dokumentow(n: int) -> str:
    if n == 1:
        return "1 dokument"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} dokumenty"
    return f"{n} dokumentów"


licznik = st.empty()

ready = backend.model_available()
if not ready:
    st.warning(f"Model niedostępny. Uruchom Ollamę i `ollama pull {backend.MODEL}`.")

with st.expander("Dodaj nowe dokumenty"):
    uploads = st.file_uploader("Wybierz pliki PDF", type="pdf", accept_multiple_files=True)
    for f in uploads or []:
        if backend.save_upload(f.name, f.getvalue()):
            st.success(f"Dodano: {f.name}")
        else:
            st.error(f"{f.name} nie jest poprawnym plikiem PDF.")

licznik.caption(f"W kolekcji: {liczba_dokumentow(len(backend.list_documents()))} PDF")

question = st.text_input("Twoje pytanie", placeholder="Np. Czym jest anomalia Belady'ego?")

if st.button("Zapytaj", type="primary", disabled=not question or not ready):
    if not backend.index_is_current():
        with st.spinner("Indeksowanie dokumentów (przy pierwszym uruchomieniu lub po zmianie plików)…"):
            n = backend.build_index()
        st.info(f"Zaindeksowano {n} fragmentów.")
    with st.spinner("Szukam odpowiedzi i weryfikuję ją ze źródłami…"):
        result = backend.answer(question)

    st.subheader("Odpowiedź")
    if result.found:
        st.write(result.text)
    else:
        st.info(result.text)
    for w in result.warnings:
        st.warning(w)
    if result.sources:
        st.subheader("Źródła")
        for s in result.sources:
            with st.expander(f"[{s.number}] {s.document}, s. {s.page} (trafność {s.score:.2f})"):
                st.write(s.text)
