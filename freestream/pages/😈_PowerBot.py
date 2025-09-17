import math
import os
from datetime import datetime

import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from pages import set_bg_local, message_background_shading, search_chat_history, calculate, clock, env_loader
from serpapi import GoogleSearch

st.set_page_config(
    page_title="FreeStream: PowerBot", page_icon="😈"
)

# Decorate the page
st.title("FreeStream")
st.header(":green[_PowerBot has a toolkit and reasons before responding_]", divider="red")

# Check for API keys and environment variables
env_loader()

# Sidebar
st.sidebar.subheader("__User Panel__")

# Button to clear conversation history
if st.sidebar.button("Clear message history", width="stretch"):
    st.session_state.clear()

# Checkbox to activate Reasoner in place of Chatter
st.sidebar.toggle(
    label="Reasoner",
    key="use_reasoner",
    help="Toggle ON to use DeepSeek-Reasoner for the final *streamed* output. Otherwise, use DeepSeek-Chat."
)

# Add the sidebar temperature slider
st.sidebar.markdown(" ### Temperature Slider")
temperature_slider = st.sidebar.slider(
    label=""":orange[Set chosen LLM(reasoner/chat) temperature]. The :blue[lower] the temperature, the :blue[less] random the model will be. The :blue[higher] the temperature, the :blue[more] random the model will be.""",
    min_value=0.2,
    max_value=1.0,
    value=0.5,
    step=0.01,
    key="temperature_slider",
)

# Define a GIF toggle
gif_bg = st.sidebar.toggle(
    label="Custom Background",
    value=False,
    key="gif_background",
    help="Turn on an experimental background.",
)
if gif_bg:
    st.sidebar.file_uploader(
        label="Upload custom background",
        type=["jpg", "jpeg", "png", "gif", "bmp"],
        accept_multiple_files=False,
        key="uploaded_background",
        help="Upload a picture to change the background of the chatbot. You may upload a JPG, PNG, or GIF.",
    )
    if st.session_state.uploaded_background:
        set_bg_local(st.session_state.uploaded_background)
    else:
        set_bg_local("assets/62.gif")

    if st.sidebar.checkbox(label="Message Shading"):
        st.markdown(message_background_shading, unsafe_allow_html=True)

# Initialize session state for messages and thoughts
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thoughts" not in st.session_state:
    st.session_state.thoughts = []

# Display chat history (excluding system messages)
for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if st.secrets.DEEPSEEK.DEEPSEEK_API_KEY:
    # Response model
    reasoner_model = ChatDeepSeek(
    temperature=temperature_slider,  # High(er) temperature for creativity
    api_key=st.secrets.DEEPSEEK.DEEPSEEK_API_KEY,
    model="deepseek-reasoner",
    max_tokens=8192,
    )

    # Response model
    chatter_model = ChatDeepSeek(
    temperature=temperature_slider,  # High(er) temperature for creativity
    api_key=st.secrets.DEEPSEEK.DEEPSEEK_API_KEY,
    model="deepseek-chat",
    max_tokens=8192,
    )
    # Agent model
    # Because deepseek-reasoner cannot use tools, we use the "deepseek-chat" model for our agent
    agent_model = ChatDeepSeek(
    temperature=0.01,  # Low temperature for straightforward decision making
    api_key=st.secrets.DEEPSEEK.DEEPSEEK_API_KEY,
    model="deepseek-chat",
    max_tokens=8192,
    )

    # First, create the output formatting function
    def format_weather_data(text_blocks):
        """Format the weather data from text_blocks into a readable string"""
        formatted_text = ""
        
        for block in text_blocks:
            if block["type"] == "paragraph":
                formatted_text += f"{block['snippet']}\n\n"
            elif block["type"] == "heading":
                formatted_text += f"--- {block['snippet']} ---\n"
            elif block["type"] == "list" and "list" in block:
                for item in block["list"]:
                    formatted_text += f"• {item['snippet']}\n"
                formatted_text += "\n"
        
        return formatted_text.strip()

    # Define the tool using the langchain decorator
    @tool
    def get_weather(location: str) -> str:
        """Get current weather information for a specific location. Expects a location, and "weather" is always appended to the query.
        Example: "portland oregon", "new york", "london uk"
        """
        try:
            params = {
                "engine": "google_ai_mode",
                "q": f"{location} weather",
                "api_key": st.secrets.SERPAPI.SERPAPI_KEY,
                "location": "Portland, OR"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "text_blocks" in results and len(results["text_blocks"]) > 0:
                # Extract and format weather information from the text blocks
                weather_info = format_weather_data(results["text_blocks"])
                return f"Weather in {location}:\n{weather_info}"
            else:
                return f"Could not find weather information for {location}"
                
        except Exception as e:
            return f"Error fetching weather data: {str(e)}"

    @tool
    def web_search(query: str) -> str:
        """Search the web using Google. Pass in the query you want to search. You can use anything that you would use in a regular Google search. e.g. inurl:, site:, intitle:. 
        """
        try:
            params = {
                "engine": "google_ai_mode",
                "q": f"{query}",
                "api_key": st.secrets.SERPAPI.SERPAPI_KEY,
                "location": "Portland, OR"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "text_blocks" in results and len(results["text_blocks"]) > 0:
                # Extract and format weather information from the text blocks
                output = results["text_blocks"]
                return f"{query} results:\n{output}"
            else:
                return f"Could not find search Google for {query}"
                
        except Exception as e:
            return f"Error fetching weather data: {str(e)}"

    # Create agent with tools and memory
    agent = create_agent(
        agent_model,
        tools=[get_weather, web_search, clock, calculate, search_chat_history],
        prompt=SystemMessage(content="You are preparing information for another AI assistant. Use your tools as necessary to complete assigned tasks and address user question(s).")
    )

# Handle user input
if user_input := st.chat_input("type here<3"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if st.secrets.DEEPSEEK.DEEPSEEK_API_KEY:
        # Convert messages to LangChain format
        lc_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # Step 1: Agent phase - generate reasoning
        with st.chat_message("agent"):
            agent_placeholder = st.empty()
            agent_placeholder.markdown("🤔 Thinking...")
            
            try:
                agent_response = agent.invoke({"messages": lc_messages})
                reasoning = agent_response["messages"][-1].content
                agent_placeholder.markdown(f"🤔 Finished Thinking: {reasoning}")
                st.session_state.thoughts.append(reasoning)
            except Exception as e:
                error_msg = f"Error in thinking process: {str(e)}"
                agent_placeholder.markdown(f"❌ {error_msg}")
                st.session_state.thoughts.append(error_msg)
                reasoning = "I encountered an error while processing your request."
        
        # Step 2: Chatter phase - generate response based on reasoning
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Prepare chatter messages with reasoning
            chatter_messages = [
                SystemMessage(content=f"Based on this reasoning: {reasoning}, generate a helpful response to the user."),
                *lc_messages
            ]
            
            # Select model based on `st.session_state.use_reasoner`
            if st.session_state.use_reasoner:
                generative_model = reasoner_model
            else:
                generative_model = chatter_model
            
            # Attempt streaming response
            try:
                for chunk in generative_model.stream(chatter_messages):
                    if hasattr(chunk, 'content'):
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.error("Please enter your DeepSeek API key in the sidebar.")

# st.write(st.session_state.messages)