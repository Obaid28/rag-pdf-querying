from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
import tempfile
import re
import os

app = FastAPI(title="RAG PDF Query API")

# Enable CORS for local testing or frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility: Clean string content
def clean_string(input_string):
    cleaned_string = re.sub(r'[^\w\s]', '', input_string)
    cleaned_string = cleaned_string.replace('\n', ' ')
    cleaned_string = cleaned_string.replace('•', '')
    return cleaned_string

# Utility: Get source metadata from documents
def get_metadata_for_sources(sources):
    metadata = []
    for idx, doc in enumerate(sources):
        meta = doc.metadata.copy()
        meta["chunk_index"] = idx
        metadata.append(meta)
    return metadata

@app.post("/query/")
async def query_pdf(pdf: UploadFile = File(...), question: str = Form(...)):
    # Step 1: Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await pdf.read())
        temp_path = temp_pdf.name

    try:
        # Step 2: Load PDF using LangChain
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Step 3: Clean + prepare text
        texts = [clean_string(doc.page_content) for doc in documents]
        full_text = " ".join(texts)

        # Step 4: Semantic chunking
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = text_splitter.split_text(full_text)

        # Step 5: Embed chunks using HuggingFace
        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

        # Step 6: Language model setup with prompt tuning
        prompt_prefix = (
            "You are a helpful assistant specialized in extracting factual answers from domain-specific PDF documents.\n"
            "If the question is ambiguous or unrelated to the document, respond appropriately.\n\n"
            f"User Query: {question}\nAnswer:"
        )
        hf_pipeline = pipeline("text-generation", model="gpt2-xl", max_new_tokens=100)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Step 7: RAG chain with retriever
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_type="similarity"),
            return_source_documents=True,
        )

        # Step 8: Query the chain
        response = qa_chain({"question": prompt_prefix, "chat_history": []})
        answer = response["answer"]
        sources = get_metadata_for_sources(response["source_documents"])

        return JSONResponse(
            content={"answer": answer.strip(), "source_chunks": sources},
            status_code=200
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        os.remove(temp_path)
