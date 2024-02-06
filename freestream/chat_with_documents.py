import logging
import sys
import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Set up page config
st.set_page_config(page_title="FreeStream: Chat with Documents", page_icon="🗣️📄")
st.title("FreeStream")
st.header(":rainbow[_Use AI as the Voice of Your Documents_]", divider="red")
st.caption(":violet[_Upload your files and watch the magic happen!_]")


@st.cache_resource(ttl="1h")  # Cache the resource
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
    # log
    logger.info("Split %s documents into %s chunks", len(docs), len(chunks))

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 10, "fetch_k": 15}
    )

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
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


st.sidebar.subheader("User Panel")

openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key below",
    type="password",
    placeholder="Expected format: 'sk-...'",
)
if not openai_api_key.startswith("sk-"):
    st.info("Please add your OpenAI API key to use GPT-3.5 Turbo.")
    st.sidebar.caption(
        "Sign up on [OpenAI's website](https://platform.openai.com/signup) and [click here to get your own API key](https://platform.openai.com/account/api-keys)."
    )
    #st.stop()
    
huggingfacehub_api_token = st.sidebar.text_input(
    "Enter your HuggingFace Hub API Token below:",
    type="password",
    #placeholder="Expected format: 'sk-...'",
)
if not huggingfacehub_api_token:
    st.info("Please add your HuggingFace Token to use Zephyr.")
    #st.sidebar.caption("Sign up on [OpenAI's website](https://platform.openai.com/signup) and [click here to get your own API key](https://platform.openai.com/account/api-keys).")
    st.stop()

uploaded_files = st.sidebar.file_uploader(
    label="Upload a PDF or text file",
    type=["pdf", "doc", "docx", "txt"],
    help="Types supported: pdf, doc, docx, txt",
    accept_multiple_files=True,
)
if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)

# Initialize Zephyr
hfllm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 512,
                "top_k": 30,
                "temperature": 0.1,
                "repetition_penalty": 1.03,
            },
            huggingfacehub_api_token=huggingfacehub_api_token,
    )
# Create a dictionary with keys to chat model classes
model_names = {
    "ChatOpenAI GPT-3.5 Turbo": ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        openai_api_key=openai_api_key,
        temperature=0.3,
        streaming=True,
    ),
    "Zephyr-7b-beta": ChatHuggingFace(llm=hfllm),
}


# Define a callback function for when a model is selected
def set_llm():
    # Set the model in session state
    st.session_state.llm = model_names[selected_model]

    # Show an alert based on what model was selected
    if st.session_state.model_selector == "Zephyr-7b-beta":
        st.warning(body="Switched to Zephyr-7b-beta!", icon="⚠️")
    elif st.session_state.model_selector == "ChatOpenAI GPT-3.5 Turbo":
        st.warning(body="Switched to ChatGPT 3.5-Turbo!", icon="⚠️")
    else:
        st.warning(
            body="Failed to change model! \nPlease contact the website builder.",
            icon="⚠️",
        )


selected_model = st.selectbox(
    label="Choose your chat model:",
    options=list(model_names.keys()),
    key="model_selector",
    on_change=set_llm,
)

# Load the selected model dynamically
llm = model_names[selected_model]

# Create a chain that ties everything together
qa_chain = ConversationalRetrievalChain.from_llm(
    # switch to 
    # create_history_aware_retriever
    llm, retriever=retriever, memory=memory, verbose=True
)

# if the length of messages is 0, or when the user \
# clicks the clear button
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
        if "messages" not in st.session_state:
            st.session_state.messages = []

        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(
            user_query, callbacks=[retrieval_handler, stream_handler]
        )