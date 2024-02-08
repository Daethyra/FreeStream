import logging
import sys
import os
import tempfile
import streamlit as st
import torch
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FreeStream-v1"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.LANGCHAIN.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = st.secrets.LANGCHAIN.LANGCHAIN_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Set up page config
st.set_page_config(page_title="FreeStream: Free AI Tooling", page_icon="🗣️📄")
st.title("FreeStream")
st.header(":rainbow[_Empowering Everyone with Advanced AI Tools_]", divider="red")
st.caption(":violet[_Democratizing access to advanced AI tools like GPT-3.5-turbo, offering a free service to simplify document retrieval and generation._]")


@st.cache_resource(ttl="1h") # Cache the resource
def configure_retriever(uploaded_files):
    """
    Configure retriever using uploaded files and return a retriever object.
    
    Args:
    - uploaded_files: List of uploaded files
    
    Returns:
    - retriever: Retriever object
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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs)
    vectordb = FAISS.from_documents(chunks, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 15})
    
    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        """
        on_retriever_start function updates status and writes a question and context retrieval label.
        
        Args:
            serialized (dict): The serialized data.
            query (str): The query string.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        """
        Perform an action when the retriever process ends.

        :param documents: list of documents
        :param kwargs: additional keyword arguments
        :return: None
        """
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

st.sidebar.subheader("__User Panel__")

openai_api_key = st.secrets.OPENAI.openai_api_key

uploaded_files = st.sidebar.file_uploader(
    label="Upload a PDF or text file",
    type=["pdf", "doc", "docx", "txt"],
    help="Types supported: pdf, doc, docx, txt",
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Create a dictionary with keys to chat model classes
model_names = {
    "ChatOpenAI GPT-3.5 Turbo": ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        openai_api_key=openai_api_key,
        temperature=0.7,
        streaming=True
    ),
}

# Define a callback function for when a model is selected
def set_llm():
    # Set the model in session state
    st.session_state.llm = model_names[selected_model]
    
    # Show an alert based on what model was selected
    if st.session_state.model_selector == model_names["ChatOpenAI GPT-3.5 Turbo"]:
        st.warning(body="Switched to ChatGPT 3.5-Turbo!", icon="⚠️")
    else:
        st.warning(body="Failed to change model! \nPlease contact the website builder.", icon="⚠️")

selected_model = st.selectbox(
    label="Choose your chat model:",
    options=list(model_names.keys()),
    key="model_selector",
    on_change=set_llm
)

# Load the selected model dynamically
llm = model_names[selected_model]

# Create a chain that ties everything together
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    verbose=True
)

# if the length of messages is 0, or when the user \
    # clicks the clear button,
    # show a default message from the AI
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    # show a default message from the AI
    msgs.add_ai_message("How can I help you?")

# Display coversation history window
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# Display user input field and enter button
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)
            
    # Display assistant response
    with st.chat_message("assistant"):
        # Check for the presence of the "messages" key in session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        st.toast('Success!', icon="✅")