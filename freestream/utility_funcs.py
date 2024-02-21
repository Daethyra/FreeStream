import os
import sys
import logging
import tempfile
import torch
import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set up logging
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
logger = logging.getLogger(__name__)


@st.cache_resource(ttl="1h")  # Cache the resource
def configure_retriever(uploaded_files):
    """
    This function configures and returns a retriever object for a given list of uploaded files.

    The function performs the following steps:
    1. Reads the documents from the uploaded files.
    2. Splits the documents into smaller chunks.
    3. Creates embeddings for each chunk using the HuggingFace's MiniLM model.
    4. Stores the embeddings in a FAISS vector database.
    5. Defines a retriever object that uses the FAISS vector database to search for similar documents.

    Args:
        uploaded_files (list): A list of Streamlit uploaded file objects.

    Returns:
        retriever (Retriever): A retriever object that can be used to search for similar documents.
    """
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = UnstructuredFileLoader(temp_filepath)
        docs.extend(loader.load())
        logger.info("Loaded document: %s", file.name)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=75)
    chunks = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    # quickly create a GPU detection line for model_kwargs
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs
    )
    vectordb = FAISS.from_documents(chunks, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 7}
    )

    return retriever

# Define a callback function for when a model is selected
def set_llm():
    """
    Updates the large language model (LLM) in the session state based on the user's selection.
    Also, displays an alert based on the selected model.

    Parameters:
    - None

    Returns:
    - None

    This function has the following side effects:
    1. It updates the `llm` key in the `st.session_state` dictionary with the selected model.
    2. It displays a warning message when the user switches to the "ChatOpenAI GPT-3.5 Turbo" model.
    3. It displays a failure warning message when the user fails to change the model (e.g., due to unsupported models).
    """
    # Set the model in session state
    st.session_state.llm = model_names[selected_model]

    # Show an alert based on what model was selected
    if st.session_state.model_selector == model_names["ChatOpenAI GPT-3.5 Turbo"]:
        st.warning(body="Switched to ChatGPT 3.5-Turbo!", icon="⚠️")

    # Add more if statements for each added model
    # if st.session_state.model_selector == model_names["GPT-4"]:
    #     ...
    else:
        st.warning(
            body="Failed to change model! \nPlease contact the website builder.",
            icon="⚠️",
        )