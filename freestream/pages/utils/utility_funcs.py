import logging
import os
import sys
import tempfile
from typing import List

import requests
import streamlit as st
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import \
    StreamlitChatMessageHistory
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class RetrieveDocuments:
    """
    A class for retrieving and managing documents for processing.

    This class is responsible for loading documents from uploaded files, splitting them into chunks,
    generating embeddings for these chunks, and configuring a retriever for efficient document retrieval.

    Attributes:
        docs (list): A list of documents loaded from the uploaded files.
        temp_dir (TemporaryDirectory): A temporary directory for storing uploaded files.
        text_splitter (RecursiveCharacterTextSplitter): An instance of a text splitter for dividing documents into chunks.
        vectordb (FAISS): A vector database for storing embeddings and facilitating document retrieval.
        retriever (Retriever): A configured retriever for retrieving documents based on embeddings.
        embeddings (HuggingFaceEmbeddings): An instance for generating embeddings for document chunks.
    """

    def __init__(self):
        """
        Initialize the RetrieveDocuments class with a list of uploaded files.

        Args:
            None
        """
        self.docs = []
        self.temp_dir = tempfile.TemporaryDirectory()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, chunk_overlap=50
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

    @st.cache_resource(ttl="1h")
    def configure_retriever(_self, uploaded_files: List[Document]):
        """
        Configure the retriever by creating a vector database from the provided chunks and embeddings.

        This method first splits the loaded documents into chunks using the text splitter.
        It then generates embeddings for these chunks and creates a vector database (FAISS)
        from these chunks and embeddings.
        Finally, it configures a retriever using the vector database and returns the
        configured retriever.
        
        Args:
            uploaded_files (List[Document]): A list of documents loaded from the uploaded files.

        Returns:
            Retriever: A configured retriever for retrieving documents based on embeddings.
        """

        # Read documents
        docs = _self.docs
        temp_dir = _self.temp_dir
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = UnstructuredFileLoader(temp_filepath)
            docs.extend(loader.load())
            logger.info("Loaded document: %s", file.name)

        # Split documents
        text_splitter = _self.text_splitter
        chunks = text_splitter.split_documents(docs)

        vectordb = FAISS.from_documents(chunks, _self.embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "fetch_k": 14}
        )

        return retriever


# Define a callback function for selecting a model
def set_llm(selected_model: str, model_names: dict):
    """
    Sets the large language model (LLM) in the session state based on the user's selection.
    Also, displays an alert based on the selected model.
    """
    try:
        # Set the model in session state
        st.session_state.llm = model_names[selected_model]

        # Show an alert based on what model was selected
        st.success(body=f"Switched to {selected_model}!", icon="✅")

    except Exception as e:
        # Log the detailed error message
        logging.error(
            f"Unsupported model selected or Error changing model: {e}\n{selected_model}"
        )
        # Display a more informative error message to the user
        st.error(f"Failed to change model! Error: {e}\n{selected_model}")


# Dictionary of model options and their corresponding file URLs
upscaler_model_options = {
    # "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus (Quality)": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    # "RealESRNet_x4plus (Faster)": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B (Anime Input ONLY)": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    # "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    # "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}


# Define a function to download a model
def download_model(
    model_name: str, file_url: str, models_dir: str = "freestream/pages/utils/realesrgan/models"
):
    """
    Downloads a specified model from a given URL if it's not already present in the local directory.

    Parameters:
    - model_name (str): The name of the model to download. This is used to construct the local file path.
    - file_url (str): The URL from which to download the model file.

    Returns:
    - None
    """

    # Check if the models directory exists. If it doesn't, create it.
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info("Created models directory: %s", models_dir)

    # Construct the local file path for the model.
    model_path = os.path.join(models_dir, f"{model_name}.pth")

    # Check if the model file already exists in the local directory.
    if not os.path.isfile(model_path):
        # Use st.spinner to show a spinner while the download is in progress.
        with st.spinner(f"Downloading {model_name}..."):
            # Send a GET request to the provided file URL to download the model file.
            response = requests.get(file_url)

            # Open the local file path in write-binary mode ('wb').
            with open(model_path, "wb") as f:
                # Write the content of the response to the local file.
                f.write(response.content)
                logger.info("\nDownloaded model: %s", model_name)

        # Use st.success to show a success message once the download is complete.
        st.toast(f"{model_name} downloaded successfully!", icon="✅")

    else:
        # If the model file already exists, inform the user.
        st.toast(f"{model_name} is already downloaded and ready to use.")
        logger.info("Model %s already exists in %s", model_name, models_dir)


class StreamHandler(BaseCallbackHandler):
    """
    A callback handler for streaming the model's output to the user interface.

    This handler updates the user interface with the model's token by token. It also ignores the rephrased question    as output by using a run ID.

    Attributes:
        container (DeltaGenerator): The delta generator object for updating the user interface.
        text (str): The text that has been generated by the model.
        run_id_ignore_token (str): The run ID for ignoring the rephrased question as output.
    """

    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        """
        Initialize the StreamHandler object.

        Args:
            container (DeltaGenerator): The delta generator object for updating the user interface.
            initial_text (str): The initial text for the user interface.
        """
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        """
        Called when the language model starts generating a response.

        This method sets the run ID for ignoring the rephrased question as output.

        Args:
            serialized (dict): The serialized data for the language model.
            prompts (list): The list of prompts for the language model.
            kwargs: Additional keyword arguments.
        """
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Called when the language model generates a new token.

        This method updates the user interface with the new token and appends it to the text.

        Args:
            token (str): The new token generated by the language model.
            kwargs: Additional keyword arguments.
        """
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    """
    A callback handler for printing the context retrieval status.

    This handler updates the status of the retrieval process, including the question, document sources,
    and page contents. It also changes the status label and state according to the retrieval process.

    Attributes:
        container (Container): The container object that contains the status object.
        status (Status): The status object for updating the retrieval process status.
    """

    def __init__(self, container):
        """
        Initialize the PrintRetrievalHandler object.

        Args:
            container (Container): The container object that contains the status object.
        """
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        """
        Called when the retriever starts the retrieval process.

        This method writes the question to the status and updates the label of the status.

        Args:
            serialized (dict): The serialized data for the retrieval process.
            query (str): The question for which the context is being retrieved.
            kwargs: Additional keyword arguments.
        """
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        """
        Called when the retriever finishes the retrieval process.

        This method prints the document sources and page contents to the status and updates the state of the status.

        Args:
            documents (list): The list of documents retrieved for the question.
            kwargs: Additional keyword arguments.
        """
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")
