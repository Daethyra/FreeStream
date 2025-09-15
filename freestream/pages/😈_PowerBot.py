import math
import os
from datetime import datetime

import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from serpapi import GoogleSearch

st.set_page_config(
    page_title="FreeStream: PowerBot", page_icon="😈"
)

# Decorate the page
st.title("FreeStream")
st.header(":green[_PowerBot has a toolkit and reasons before responding_]", divider="red")

# Check for DeepSeek API Key before continuing
if "DEEPSEEK_API_KEY" in st.secrets.DEEPSEEK:
    DEEPSEEK_API_KEY = st.secrets.DEEPSEEK.DEEPSEEK_API_KEY
else:
    DEEPSEEK_API_KEY = st.sidebar.text_input("DeepSeek API Key", type="password")

if "SERPAPI_KEY" in st.secrets.SERPAPI:
    SERPAPI_KEY = st.secrets.SERPAPI.SERPAPI_KEY
else:
    SERPAPI_KEY = st.sidebar.text_input("Serp API Key", type="password")

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FreeStream"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.LANGCHAIN.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = st.secrets.LANGCHAIN.LANGCHAIN_API_KEY

# Sidebar
st.sidebar.subheader("__User Panel__")
# Add the sidebar temperature slider
st.sidebar.markdown(" ### Temperature Slider")
temperature_slider = st.sidebar.slider(
    label=""":orange[Set LLM Temperature]. The :blue[lower] the temperature, the :blue[less] random the model will be. The :blue[higher] the temperature, the :blue[more] random the model will be.""",
    min_value=0.5,
    max_value=1.0,
    value=0.7,
    step=0.01,
    key="temperature_slider",
)

# Button to clear conversation history
if st.sidebar.button("Clear message history", use_container_width=True):
    st.session_state.clear()

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

if DEEPSEEK_API_KEY:
    thinker_model = ChatDeepSeek(
    temperature=0.1,  # Low temperature for logical thinking
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    #max_tokens=256,
    streaming=False
    )

    chatter_model = ChatDeepSeek(
    temperature=temperature_slider,  # Higher temperature for creative responses
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    #max_tokens=512,
    streaming=True
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
        """Get current weather information for a specific location.
        Example: "portland oregon", "new york", "london uk"
        """
        try:
            params = {
                "engine": "google_ai_mode",
                "q": f"{location} weather",
                "api_key": SERPAPI_KEY,
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
                "api_key": SERPAPI_KEY,
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

    @tool
    def clock():
        """Get the datetime. Returns datetime.now().strftime("%Y-%m-%d %I:%M %p")"""    
        return datetime.now().strftime("%Y-%m-%d %I:%M %p")

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression. You can use basic operators (+, -, *, /, ^) and functions like sqrt, sin, cos, etc.
        Example expressions: 
        - "2 + 3 * 4" 
        - "sqrt(16)" 
        - "sin(30) + cos(60)"
        """
        try:
            # Replace ^ with ** for exponentiation
            expression = expression.replace('^', '**')
            
            # Add math functions to the evaluation context
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 
                'tan': math.tan, 'log': math.log, 'log10': math.log10,
                'pi': math.pi, 'e': math.e
            }
            
            # Evaluate the expression safely
            result = eval(expression, {"__builtins__": None}, safe_dict)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    # Create agent with tools and memory
    agent = create_agent(
        chatter_model,
        tools=[get_weather, web_search, clock, calculate],
        prompt=SystemMessage(content="You are assisting the user with whatever they need. Use your tools as necessary to complete assigned tasks and answer user questions.")
    )

# Handle user input
if user_input := st.chat_input("type here<3"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if DEEPSEEK_API_KEY:
        # Convert messages to LangChain format
        lc_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # Step 1: Thinker phase - generate reasoning
        with st.chat_message("thinker"):
            thinker_placeholder = st.empty()
            thinker_placeholder.markdown("🤔 Thinking...")
            
            try:
                thinker_response = agent.invoke({"messages": lc_messages})
                reasoning = thinker_response["messages"][-1].content
                thinker_placeholder.markdown(f"🤔 Finished Thinking: {reasoning}")
                st.session_state.thoughts.append(reasoning)
            except Exception as e:
                error_msg = f"Error in thinking process: {str(e)}"
                thinker_placeholder.markdown(f"❌ {error_msg}")
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
            
            # Stream the response from chatter model
            try:
                for chunk in chatter_model.stream(chatter_messages):
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