import os

import streamlit as st
from langchain import hub
from langchain.agents import (AgentExecutor,
                              create_conversational_retrieval_agent)
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import \
    StreamlitChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from pages.utils import chat_models, footer

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FS_Agent-v4.0.0"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.LANGCHAIN.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = st.secrets.LANGCHAIN.LANGCHAIN_API_KEY

### Streamlit Page Config ###
# Set up Streamlit app
st.set_page_config(page_title="TavilySearch Chatbot")

# Show footer
st.markdown(footer, unsafe_allow_html=True)

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

### ----------------------------------------------------- ###


### LLM Peripherals setup ###
# Set up memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)

# Pull a prompt from the LangChain Hub
hub_prompt = hub.pull("hwchase17/openai-tools-agent")

## TOOLS ##
# Define TavilySearch tool
tavily_search = TavilySearchResults(
    TAVILY_API_KEY=st.secrets.TAVILY.TAVILY_API_KEY, max_results=5
)

# Define tools
tools = [tavily_search]
## TOOLS ##

### ----------------------------------------------------- ###


### Agent Configuration ###
# Load the selected model dynamically
llm = model_names[
    selected_model
]  # Get the selected model from the `model_names` dictionary


# Initialize the agent with the tool and memory
agent = create_conversational_retrieval_agent(
    tools=[tavily_search],
    llm=llm,
    prompt=hub_prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    callbacks=st_callback,
    
)

### ----------------------------------------------------- ###

# Streamlit app
if prompt := st.chat_input("Ask a question"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = agent_executor.run(input=prompt, callbacks=[st_callback])
        st.write(response)
