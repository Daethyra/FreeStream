import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import LLMMathChain
from utility_funcs import (
    configure_retriever,
    set_llm,
    llm_math,
    tav_search,
    vectorstore_tool,
    tools,
)

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FreeStream-v2.1"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.LANGCHAIN.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = st.secrets.LANGCHAIN.LANGCHAIN_API_KEY

# Load tool API keys
os.environ["TAVILY_API_KEY"] = st.secrets.TAVILY.TAVILY_API_KEY

# Set up page config
st.set_page_config(page_title="FreeStream: Free AI Tooling", page_icon="🗣️📄")
st.title("FreeStream")
st.header(":green[_Welcome_]", divider="red")
st.caption(":violet[_General purpose chatbot assistant_]")
st.sidebar.subheader("__User Panel__")

# Add a way to upload files
uploaded_files = st.sidebar.file_uploader(
    label="Upload a PDF or text file",
    type=["pdf", "doc", "docx", "txt"],
    help="Types supported: pdf, doc, docx, txt",
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory(key="chat_history")

# Create a dictionary with keys to chat model classes
model_names = {
    "ChatOpenAI GPT-3.5 Turbo": ChatOpenAI(  # Define a dictionary entry for the "ChatOpenAI GPT-3.5 Turbo" model
        model_name="gpt-3.5-turbo-0125",  # Set the OpenAI model name
        openai_api_key=st.secrets.OPENAI.openai_api_key,  # Set the OpenAI API key from the Streamlit secrets manager
        temperature=0.7,  # Set the temperature for the model's responses
        streaming=True,  # Enable streaming responses for the model
    ),
}

# Create a dropdown menu for selecting a chat model
selected_model = st.selectbox(
    label="Choose your chat model:",  # Set the label for the dropdown menu
    options=list(model_names.keys()),  # Set the available model options
    key="model_selector",  # Set a unique key for the dropdown menu
    on_change=set_llm,  # Set the callback function
)

# Load the selected model dynamically
llm = model_names[
    selected_model
]  # Get the selected model from the `model_names` dictionary

# Initialize Callbacks for Agent
st_callback = StreamlitCallbackHandler(st.container())
### Initialize agent ###
react_agent = create_react_agent(llm, tools, hub.pull("hwchase17/react"))
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    max_execution_time=180,
    max_iterations=7,
    memory=msgs,
    )
### ---------------- ###

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
        if "messages" not in st.session_state:
            st.session_state.messages = []

        response = agent_executor.invoke({
            "question": user_query,
            "chat_history": msgs.messages,
            "tools": tools
        }, callbacks=[st_callback])
        st.toast("Success!", icon="✅")
