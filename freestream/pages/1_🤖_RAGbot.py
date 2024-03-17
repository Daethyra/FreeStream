import os

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pages.utils import (
    PrintRetrievalHandler,
    RetrieveDocuments,
    StreamHandler,
    chat_models,
    footer,
    set_llm,
)

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FS_Ragbot-v4.0.0"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.LANGCHAIN.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = st.secrets.LANGCHAIN.LANGCHAIN_API_KEY

# Set up page config
st.set_page_config(page_title="FreeStream: RAGbot", page_icon="🤖")
st.title("🤖RAGbot")
st.header(":green[_Retrieval Augmented Generation Chatbot_]", divider="red")
st.caption(":violet[_Ask Your Documents Questions_]")
# Show footer
st.markdown(footer, unsafe_allow_html=True)

# Add sidebar
st.sidebar.subheader("__User Panel__")
# Add file-upload button
uploaded_files = st.sidebar.file_uploader(
    label="Upload a PDF or text file",
    type=["pdf", "doc", "docx", "txt"],
    help="Types supported: pdf, doc, docx, txt \n\nConsider the size of your files before you upload. Processing speed varies by server load.",
    accept_multiple_files=True,
)
if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

retriever = RetrieveDocuments().configure_retriever(uploaded_files)

# Add temperature header
temperature_header = st.sidebar.markdown(
    """
    ## Temperature Slider
    """
)
# Add the sidebar temperature slider
temperature_slider = st.sidebar.slider(
    label=""":orange[Set LLM Temperature]. The :blue[lower] the temperature, the :blue[less] random the model will be. The :blue[higher] the temperature, the :blue[more] random the model will be.""",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    key="llm_temperature",
)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)

# Create a dropdown menu for selecting a chat model
selected_model = st.selectbox(
    label="Choose your chat model:",  # Set the label for the dropdown menu
    options=list(model_names.keys()),  # Set the available model options
    key="model_selector",  # Set a unique key for the dropdown menu
    on_change=lambda: set_llm(
        st.session_state.model_selector, model_names
    ),  # Set the callback function
)

# Load the selected model dynamically
llm = model_names[
    selected_model
]  # Get the selected model from the `model_names` dictionary

# Create a chain that ties everything together
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
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
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(
            user_query, callbacks=[retrieval_handler, stream_handler]
        )
        # Force print Gemini's response
        if selected_model == "Gemini-Pro":
            st.write(response)
