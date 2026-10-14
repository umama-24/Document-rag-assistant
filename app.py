import os
import requests
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Document Q&A Assistant", layout="wide", initial_sidebar_state="expanded"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks_store" not in st.session_state:
    st.session_state.chunks_store = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# Load lightweight embedding model locally ($0 cost)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# --- Layout matching the reference image ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📚 AI Document Q&A Assistant")
    st.markdown(
        "Ask questions about your documents and get instant, accurate answers "
        "with source evidence. Powered by AI to help you extract insights from your "
        "PDF collections quickly and reliably."
    )
    st.markdown(
        "- Sales team extracting competitor insights from earnings calls\n"
        "- Legal team finding clauses across 10s of contracts\n"
        "- Investment firm analyzing portfolio reports in minutes\n"
        "- Support team finding answers in product documentation"
    )
    
    st.markdown("---")
    st.markdown("### References")
    
   
# References Display (Guaranteed Native UI - No PyArrow, No Code Blocks, No Overlap)
    if st.session_state.last_sources:
        for i, src in enumerate(st.session_state.last_sources, 1):
            with st.container(border=True):
                col_info, col_text = st.columns([1, 2])
                with col_info:
                    st.markdown(f"**PDF:** {src['source']}")
                    st.caption(f"**Page:** {src['page']}")
                with col_text:
                    clean_excerpt = src['text'][:120].replace('\n', ' ').strip()
                    st.write(f'"{clean_excerpt}..."')
    else:
        st.info("No documents queried yet. Upload PDFs and generate an answer to see references.")

with col2:
    st.markdown("### Question")
    user_query = st.text_input("What are the key AI investments and priorities mentioned?", label_visibility="collapsed", placeholder="Enter your question here...")

    st.markdown("### PDFs")
    uploaded_files = st.file_uploader(
        "Drag 'n' drop PDFs, or click to select", type=["pdf"], accept_multiple_files=True, key="pdf_uploader"
    )

    # Process PDFs if uploaded
    if uploaded_files and st.session_state.vector_store is None:
        with st.spinner("Processing documents..."):
            all_chunks = []
            
            for uploaded_file in uploaded_files:
                temp_path = f"./{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Read PDF using pdfplumber page by page
                with pdfplumber.open(temp_path) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            # Simple chunking by paragraphs/sentences or block size
                            chunk_size = 500
                            for i in range(0, len(text), chunk_size):
                                chunk_text = text[i:i + chunk_size]
                                all_chunks.append({
                                    "source": uploaded_file.name,
                                    "page": page_idx + 1,
                                    "text": chunk_text
                                })

            if all_chunks:
                st.session_state.chunks_store = all_chunks
                texts = [c["text"] for c in all_chunks]
                
                # Create embeddings & FAISS index
                embeddings = embed_model.encode(texts, show_progress_bar=False)
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(np.array(embeddings).astype("float32"))
                
                st.session_state.vector_store = index
                st.success("Documents processed successfully!")

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} PDFs selected")

    generate_btn = st.button("Generate answer", type="primary", use_container_width=True)

    if generate_btn:
        if st.session_state.vector_store is None:
            st.warning("Please upload and process at least one PDF first.")
        elif not user_query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                # Embed query and search FAISS index
                query_vector = embed_model.encode([user_query])
                k = min(3, len(st.session_state.chunks_store))
                distances, indices = st.session_state.vector_store.search(
                    np.array(query_vector).astype("float32"), k
                )

                retrieved_chunks = [st.session_state.chunks_store[idx] for idx in indices[0]]
                
                # Build context for Ollama
                context_str = "\n\n".join([c["text"] for c in retrieved_chunks])
                prompt = (
                    "You are an AI assistant for question-answering tasks. "
                    "Use the following retrieved context to answer the question concisely. "
                    "If you do not know, say that you don't know.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {user_query}\n"
                    "Answer:"
                )

                # Query local Ollama instance via HTTP API
                try:
                    res = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "mistral",
                            "prompt": prompt,
                            "stream": False
                        }
                    )

                    if res.status_code == 200:
                        st.session_state.last_answer = res.json().get("response", "No response generated.")
                        st.session_state.last_sources = retrieved_chunks
                        st.rerun()
                    else:
                        st.error("Error connecting to Ollama server.")
                except Exception as e:
                    st.error(f"Could not connect to Ollama: {e}")

    st.markdown("### Answer")
    answer_container = st.container(border=True)
    with answer_container:
        if st.session_state.last_answer:
            st.markdown(st.session_state.last_answer)
        else:
            st.markdown("_Your generated answer will appear here..._")