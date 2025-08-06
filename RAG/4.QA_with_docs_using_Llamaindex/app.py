
import streamlit as st
from src.config import load_env, get_llm, get_embedding_model
from src.data_ingestion import load_data
from src.embeddings import build_and_persist_index, load_query_engine
from llama_index.core.settings import Settings

# Load environment variables
load_env()

# Configure settings
llm = get_llm()
embed_model = get_embedding_model()
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 800
Settings.chunk_overlap = 20

# Streamlit app setup
st.set_page_config(page_title="QA with Documents")
st.title("Document Q&A System")

# Session state initialization
if 'process_counter' not in st.session_state:
    st.session_state.process_counter = 0
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None

# File uploader
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            documents = load_data(uploaded_file)
            build_and_persist_index(documents)
            st.session_state.process_counter += 1
            st.session_state.query_engine = None  # Reset to trigger reload
            st.success("Document processed successfully!")

# Load query engine with caching
@st.cache_resource()
def get_query_engine():
    return load_query_engine()

if st.session_state.process_counter > 0:
    if st.session_state.query_engine is None:
        st.session_state.query_engine = get_query_engine()

    # Question input
    question = st.text_input("Ask a question about the document:")
    if question and st.session_state.query_engine:
        with st.spinner("Generating answer..."):
            response = st.session_state.query_engine.query(question)
            st.write("**Answer:**", response.response)