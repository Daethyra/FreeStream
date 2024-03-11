import logging
import os
import sys
import tempfile
from typing import List

import streamlit as st
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from PIL import Image
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class RetrieveDocuments:
    """
    A class for retrieving and managing documents for processing.

    This class is responsible for loading documents from uploaded files, splitting them into chunks,
    generating embeddings for these chunks, and configuring a retriever for efficient document retrieval.

    Attributes:
        uploaded_files (list): A list of uploaded files to be processed.
        docs (list): A list of documents loaded from the uploaded files.
        temp_dir (TemporaryDirectory): A temporary directory for storing uploaded files.
        text_splitter (RecursiveCharacterTextSplitter): An instance of a text splitter for dividing documents into chunks.
        vectordb (FAISS): A vector database for storing embeddings and facilitating document retrieval.
        retriever (Retriever): A configured retriever for retrieving documents based on embeddings.
        embeddings (HuggingFaceEmbeddings): An instance for generating embeddings for document chunks.
    """

    def __init__(self, uploaded_files: List[Document]):
        """
        Initialize the RetrieveDocuments class with a list of uploaded files.

        Args:
            uploaded_files (list): A list of uploaded files to be processed.
        """
        self.uploaded_files = uploaded_files
        self.docs = []
        self.temp_dir = tempfile.TemporaryDirectory()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=0
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        self.vectordb = None
        self.retriever = None

    def load_documents(self):
        """
        Load documents from the uploaded files into the `docs` attribute.

        This method iterates over the uploaded files, writes them to a temporary directory,
        and then loads the documents using an UnstructuredFileLoader. The loaded documents are
        stored in the `docs` attribute.
        """
        for document in self.uploaded_files:
            temp_filepath = os.path.join(self.temp_dir.name, document.name)
            with open(temp_filepath, "wb") as f:
                f.write(document.getvalue())
            loader = UnstructuredFileLoader(temp_filepath)
            self.docs.extend(loader.load())
            logger.info("Loaded document: %s", document.name)

    @st.cache_resource(ttl="1h")
    def configure_retriever(_self):
        """
        Configure the retriever by creating a vector database from the provided chunks and embeddings.

        This method first splits the loaded documents into chunks using the text splitter.
        It then generates embeddings for these chunks and creates a vector database (FAISS)
        from these chunks and embeddings.
        Finally, it configures a retriever using the vector database and returns the
        configured retriever.

        Returns:
            Retriever: A configured retriever for retrieving documents based on embeddings.
        """

        # Load documents
        _self.load_documents()

        # Write a line to split the documents
        chunks = _self.text_splitter.split_documents(_self.docs)

        # Embed the chunks into a vector database
        _self.vectordb = FAISS.from_documents(chunks, _self.embeddings)

        # Create a retriever
        _self.retriever = _self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 12},
        )

        return _self.retriever


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


# Define a function to upscale images using HuggingFace and Torch
def image_upscaler(image: str) -> Image:
    """
    Upscales the input image using the specified model and returns the upscaled image.

    Parameters:
    image (str): The file path of the input image.

    Returns:
    Image: The upscaled image.
    """

    # Assign the image to a variable
    img = Image.open(image)

    # Create the upscale pipeline
    upscaler = pipeline(
        "image-to-image",
        model="caidas/swin2SR-classical-sr-x2-64",
        framework="pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Downsize the image via ratio to ensure it can be upscaled.
    # Otherwise, we run out of memory when a user uploads a huge image.
    # If the image is greater than 1024 in either dimension, then
    # the image is downsampled by a factor of 3.
    if img.width > 128 or img.height > 128:
        st.toast("Downsampling image...", icon="🔨")
        logger.info("\nDownsampling image...")
        img = img.resize((int(img.width * 0.15), int(img.height * 0.15)))
    else:
        img = img

    # Upscale the image
    st.toast("Upscaling image...", icon="🔨")
    logger.info("\nUpscaling image...")
    upscaled_img = upscaler(img)
    st.toast("Success!", icon="✅")

    return upscaled_img
