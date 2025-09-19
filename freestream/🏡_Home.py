import datetime
import streamlit as st
from freestream import footer

st.set_page_config(
    page_title="FreeStream: Streamlit Chatbot Template", page_icon="🏡"
)

st.title("FreeStream")
st.header(":green[_Expandable Chatbots._]", divider="red")
st.markdown(footer, unsafe_allow_html=True)

### Body content ###
st.write(
    """
    FreeStream is a collection of LangChain-based chatbots that are tuned for specific use-cases.
    
    FreeStream currently only leverages DeepSeek's models for generative content. DeepSeek's API is based on the OpenAI SDK, meaning you can easily change [`ChatDeepSeek`](https://python.langchain.com/api_reference/deepseek/chat_models/langchain_deepseek.chat_models.ChatDeepSeek.html#langchain_deepseek.chat_models.ChatDeepSeek) to [`ChatOpenAI`](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html), both of which are based on [`BaseChatOpenAI`](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.BaseChatOpenAI.html#langchain_openai.chat_models.base.BaseChatOpenAI).
    """
)

st.subheader("Pre-built bots")
st.write(
    """
    #### :blue[PowerBot]:
    
    :orange[*Flexible Agent*]
    
    PowerBot first uses an instance of a DeepSeek Agent to perform a multi-step thinking process at low temperature to analyze the query, use tools as necessary, and then provide step-by-step analysis to a second instance of DeepSeek without Agentic properties, meaning the second model simply responds to the user query based on the Agent's response.
    """
)

with st.expander(label=":violet[System Prompt:]", expanded=False):
    st.markdown(
        """
        *You are assisting the user with whatever they need. Use your tools as necessary to complete assigned tasks and answer user questions.*
        """
    )
with st.expander(label=":violet[Toolkit:]", expanded=False):
    st.markdown(
        """
        1. Get Weather: Serp API query to "google_ai_mode" using `f"{location} weather"`.
        
        2. Web Search: Serp API query to "google_ai_mode" using `f"{query}`".
        
        3. Clock: Returns `datetime.now().strftime("%Y-%m-%d %I:%M %p")`
        
        4. Calculate: Evaluate a mathematical expression. You can use basic operators (+, -, *, /, ^) and functions like sqrt, sin, cos, etc.
        Example expressions: 
            - "2 + 3 * 4" 
            - "sqrt(16)" 
            - "sin(30) + cos(60)"
        """
    )

st.write(
    """
    #### :blue[MathBot]:
    
    :orange[*The math-only precursor to PowerBot*]
    
    MathBot works in an identical way to :blue[PowerBot], but only has a calculator tool.
    """
)

with st.expander(label=":violet[System Prompt:]", expanded=False):
    st.markdown(
        """
        You are a thinker agent. Analyze the user's message and provide step-by-step reasoning. Use the calculator tool when mathematical calculations are needed. Be concise and logical.
        """
    )
st.divider()

st.sidebar.markdown(
    """
    ### References
    
    * **[Run This App On Your Own Computer](https://github.com/Daethyra/FreeStream/blob/streamlit/README.md#installation)**
    * **[Platform Privacy Policies](https://github.com/Daethyra/FreeStream/blob/streamlit/README.md#llm-providers-privacy-policies)**
    * **[FreeStream's GitHub Repository](https://github.com/Daethyra/FreeStream)**
    """
)