import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import LLMMathChain
from langchain import hub
from utility_funcs import (
    configure_retriever,
    VectorStoreRetrievalTool,
    StreamHandler,
    PrintRetrievalHandler,
    set_llm,
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
if uploaded_files:
    retriever = configure_retriever(uploaded_files)
#if not uploaded_files:
#    st.info("Please upload documents to continue.")
#    st.stop()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)

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

# Import custom prompt from Hub
STARTER_PROMPT = hub.pull("daethyra/freestream")

# Define few-shot prompt examples to guide *generated* search queries
tav_description = (
    """
    A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.

    Examples where curly braces indicate the need to inject context from the chat history to create an effective search query:
    | User Input | Search Query |
    | --- | --- |
    | What is this book about? | What is {book_title} about? |
    | Who is running for President this year? | Current presidential candidates |
    | Who won the Super Bowl? | Who won the Super Bowl in {year}? |
    """
)

# Define tools for the agent
llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)
tav_search = TavilySearchResults(
    description = tav_description,
    max_results = 4
)
tools = [
    Tool(
        name="Web Search",
        func=tav_search.run,
        description="Perfect for getting current information. Use specific queries related to the chat history and the user\'s current foremost concern."  
    ),
    
    Tool(
        name="Calculator",
        func=llm_math.run,
        description="Solves maths problems. Translate a math problem into an expression that can be executed using Python\'s numexpr library. Use the output of running this code to help yourself answer the user\'s foremost concern."
    ),
    Tool(
        name="VectorStore",
        func=VectorStoreRetrievalTool._run,
        description="Searches the documents the user uploaded. Input should be in the form of a question containing full context of what you\'re looking for. Include all relevant context, because we use semantic similarity searching to find relevant documents from the Database. Returns retrieved documents."
    ),
]

# Initialize agent
react_agent = create_react_agent(llm, tools, hub.pull("hwchase17/react"))
agent_executor = AgentExecutor(agent=react_agent, tools=tools)

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

        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = agent_executor.invoke({
            "question": user_query,
            "chat_history": msgs.messages,
            "tools": tools
        }
            #user_query, callbacks=[retrieval_handler, stream_handler]
        )
        st.toast("Success!", icon="✅")
