import datetime
import streamlit as st
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from freestream import footer
from pages import save_conversation_history

st.set_page_config(
    page_title="FreeStream: A basic chatbot to build on.", page_icon="🏡"
)

st.title("FreeStream")
st.header(":green[_A basic chatbot to build on._]", divider="red")
st.markdown(footer, unsafe_allow_html=True)

DEEPSEEK_API_KEY = st.sidebar.text_input("DeepSeek API Key", type="password")
# Button to clear conversation history
if st.sidebar.button("Clear message history", use_container_width=True):
    st.session_state.clear()


# Initialize chat model (but only if API key is provided)
if DEEPSEEK_API_KEY:
    chat_model = ChatDeepSeek(
        temperature=0.5,
        api_key=DEEPSEEK_API_KEY,
        model="deepseek-chat",
        max_tokens=128,
        streaming=True
    )

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Save the formatted conversation history to a variable
formatted_history = save_conversation_history(st.session_state.messages)
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Create a sidebar button to download the conversation history
st.sidebar.download_button(
    label="Download conversation history",
    data=formatted_history,
    file_name=f"conversation_history {current_time}.md",
    mime="text/markdown",
    key="download_conversation_history_button",
    help="Download the conversation history as a text file with some formatting.",
    use_container_width=True,   
)

# Handle user input
if user_input := st.chat_input("type here<3"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
        
        # Stream the response
        for chunk in chat_model.stream(lc_messages):
            if hasattr(chunk, 'content'):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})