import os

import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import \
    StreamlitChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pages.utils import RetrieveDocuments, footer

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FS_Agent-v4.0.0"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.LANGCHAIN.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = st.secrets.LANGCHAIN.LANGCHAIN_API_KEY
os.environ["TAVILY_API_KEY"] = st.secrets.TAVILY.tavily_api_key

### Streamlit Page Config ###
# Set up Streamlit app
st.set_page_config(page_title="TavilySearch Chatbot")

# Show footer
st.markdown(footer, unsafe_allow_html=True)

# Add file-upload button
uploaded_files = st.sidebar.file_uploader(
    label="Upload a PDF or text file",
    type=["pdf", "doc", "docx", "txt"],
    help="Types supported: pdf, doc, docx, txt \n\nConsider the size of your files before you upload. Processing speed varies by server load.",
    accept_multiple_files=True,
)

## TOOLS ##
# Define TavilySearch tool
tavily_search = TavilySearchResults(max_results=5)

# Define tools based on available files
if uploaded_files:
    retriever = RetrieveDocuments.configure_retriever(uploaded_files)
    toolbox = [retriever, tavily_search]
else:
    toolbox = [tavily_search]


### Chatbot Configuration Widgets ###
# Add the sidebar temperature slider
temperature_slider = st.sidebar.slider(
    label=""":orange[Set LLM Temperature]. The :blue[lower] the temperature, the :blue[less] random the model will be. The :blue[higher] the temperature, the :blue[more] random the model will be.""",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    key="llm_temperature",
)

# Create a dictionary with keys to chat model classes
model_names = {
    "GPT-3.5 Turbo": ChatOpenAI(  # Define a dictionary entry for the "ChatOpenAI GPT-3.5 Turbo" model
        model="gpt-3.5-turbo-0125",  # Set the OpenAI model name
        openai_api_key=st.secrets.OPENAI.openai_api_key,  # Set the OpenAI API key from the Streamlit secrets manager
        temperature=temperature_slider,  # Set the temperature for the model's responses using the sidebar slider
        streaming=True,  # Enable streaming responses for the model
        max_tokens=4096,  # Set the maximum number of tokens for the model's responses
        max_retries=1,  # Set the maximum number of retries for the model
    ),
    "GPT-4 Turbo": ChatOpenAI(
        model="gpt-4-0125-preview",
        openai_api_key=st.secrets.OPENAI.openai_api_key,
        temperature=temperature_slider,
        streaming=True,
        max_tokens=4096,
        max_retries=1,
    ),
    "Claude: Haiku": ChatAnthropic(
        model="claude-3-haiku-20240307",
        anthropic_api_key=st.secrets.ANTHROPIC.anthropic_api_key,
        temperature=temperature_slider,
        streaming=True,
        max_tokens=4096,
    ),
    "Claude: Sonnet": ChatAnthropic(
        model="claude-3-sonnet-20240229",
        anthropic_api_key=st.secrets.ANTHROPIC.anthropic_api_key,
        temperature=temperature_slider,
        streaming=True,
        max_tokens=4096,
    ),
    "Claude: Opus": ChatAnthropic(
        model="claude-3-opus-20240229",
        anthropic_api_key=st.secrets.ANTHROPIC.anthropic_api_key,
        temperature=temperature_slider,
        streaming=True,
        max_tokens=4096,
    ),
    "Gemini-Pro": ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=st.secrets.GOOGLE.google_api_key,
        temperature=temperature_slider,
        top_k=50,
        top_p=0.7,
        convert_system_message_to_human=True,
        max_output_tokens=4096,
        max_retries=1,
    ),
}

# Create a dropdown menu for selecting a chat model
selected_model = st.selectbox(
    label="Choose your chat model:",  # Set the label for the dropdown menu
    options=list(model_names.keys()),  # Set the available model options
    key="model_selector",  # Set a unique key for the dropdown menu
    on_change=lambda: set_llm(
        st.session_state.model_selector, model_names
    ),  # Set the callback function
)

# Set up Streamlit callback handler
st_callback = StreamlitCallbackHandler(st.container())

### LLM Peripherals setup ###
# Set up memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)

# Pull a prompt from the LangChain Hub
hub_prompt = hub.pull("daethyra/openai-tools-agent")


### Agent Configuration ###
# Load the selected model dynamically
llm = model_names[
    selected_model
]  # Get the selected model from the `model_names` dictionary


# Initialize the agent with the tool and memory
agent = create_openai_tools_agent(
    tools=toolbox,
    llm=llm,
    prompt=hub_prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolbox,
    verbose=True,
    #callbacks=StreamlitCallbackHandler(st.container()),
)


# Streamlit app
if user_query := st.chat_input("Ask a question"):
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        response = agent_executor.invoke(
            {"input": user_query, "tool_names": toolbox, "chat_history": memory},
            callbacks=[st_callback])
        st.write(response)
