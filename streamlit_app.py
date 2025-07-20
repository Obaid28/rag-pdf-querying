import streamlit as st
import requests
import tempfile
import os

API_URL = "http://localhost:8000/query/"

st.set_page_config(page_title="RAG PDF Query", layout="wide")
st.title("📄 RAG PDF Query Interface")

st.markdown("""
Upload a PDF and ask a question.  
The system uses semantic chunking + vector embeddings (FAISS) + LLMs to answer from the document.
""")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Enter your question", placeholder="What is the main finding of this paper?")

if st.button("Ask") and uploaded_pdf and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_pdf.read())
        temp_pdf_path = temp_pdf.name

    try:
        with open(temp_pdf_path, "rb") as f:
            files = {"pdf": (uploaded_pdf.name, f, "application/pdf")}
            data = {"question": question}
            with st.spinner("Querying..."):
                response = requests.post(API_URL, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            st.success("Answer:")
            st.markdown(f"**{result['answer']}**")

            st.markdown("**Retrieved Chunks Metadata**")
            for chunk in result["source_chunks"]:
                st.json(chunk)
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Exception occurred: {str(e)}")
    finally:
        os.remove(temp_pdf_path)
