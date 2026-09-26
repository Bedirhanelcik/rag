import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Türkçe RAG Soru-Cevap")
st.title("Türkçe RAG Soru-Cevap")

with st.sidebar:
    st.header("Doküman Yükle")
    uploaded_files = st.file_uploader(
        "PDF, TXT veya MD dosyaları", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Yükle") and uploaded_files:
        files_payload = [("files", (f.name, f.getvalue())) for f in uploaded_files]
        response = requests.post(f"{API_URL}/ingest", files=files_payload)
        if response.ok:
            data = response.json()
            st.success(f"{len(data['ingested_files'])} dosya işlendi, {data['chunk_count']} chunk oluşturuldu.")
            if data["failed_files"]:
                st.warning(f"İşlenemeyen dosyalar: {', '.join(data['failed_files'])}")
        else:
            st.error(f"Yükleme başarısız: {response.text}")

question = st.text_input("Sorunuzu yazın")
if st.button("Sor") and question:
    response = requests.post(f"{API_URL}/query", json={"question": question})
    if response.ok:
        data = response.json()
        st.markdown(data["answer"])
        for source in data["sources"]:
            page_suffix = f" — sayfa {source['page_number']}" if source["page_number"] else ""
            with st.expander(f"[{source['index']}] {source['source_file']}{page_suffix}"):
                st.write(source["text"])
    else:
        st.error(f"Sorgu başarısız: {response.text}")
