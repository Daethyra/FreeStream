import base64
import os
from typing import Any, List, Union

import streamlit as st


# Define a function to change the background to an image via URL
# https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/19
def set_bg_url():
    """
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    """

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://cdn.pixabay.com/photo/2020/06/19/22/33/wormhole-5319067_960_720.jpg");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


# Define a function to change the background to a local image
# https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/16
def set_bg_local(main_bg: Union[str, st.runtime.uploaded_file_manager.UploadedFile]):
    """
    A function to set the background image from a local file or uploaded file.
    Uses caching to improve performance.

    Parameters
    ----------
    main_bg : Union[str, UploadedFile]
        Either a file path string or an UploadedFile object.
    """
    # Determine the file extension
    if isinstance(main_bg, str):
        # Extract extension from file path
        main_bg_ext = main_bg.split('.')[-1] if '.' in main_bg else "png"
    else:
        # Extract extension from uploaded file name
        if hasattr(main_bg, 'name') and '.' in main_bg.name:
            main_bg_ext = main_bg.name.split('.')[-1]
        else:
            main_bg_ext = "png"  # Default fallback
    
    # Get base64 encoded image (uses caching)
    base64_data = _get_base64_encoded_image(main_bg)

    # Set the background using Markdown
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64_data});
             background-size: cover;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner=False)
def _get_base64_encoded_image(file_path_or_uploaded_file: Union[str, st.runtime.uploaded_file_manager.UploadedFile]):
    """
    Helper function to read and encode an image file to base64.
    Cached to avoid re-reading and re-encoding the same file on every rerun.
    """
    if isinstance(file_path_or_uploaded_file, str):
        # Local file path: read the file
        with open(file_path_or_uploaded_file, "rb") as f:
            data = f.read()
    else:
        # UploadedFile object: get the data directly
        data = file_path_or_uploaded_file.getvalue()
    return base64.b64encode(data).decode()

# Create a function to save the conversation history to a file
def save_conversation_history(conversation_history: List[Any]) -> str:
    """
    Utility function to format and prepare the conversation history for download.

    Parameters:
    conversation_history (List[Any]): List of objects containing the conversation history.

    Returns:
    str: Formatted conversation history ready for download.
    """
    formatted_history = ""
    for msg in conversation_history:
        if msg.type == 'human':
            formatted_history += f"Human: {msg.content}\n\n"
        elif msg.type == 'ai':
            formatted_history += f"Assistant: {msg.content}\n\n"

    return formatted_history

@st.cache_data
def env_loader(persist=False):
    with st.sidebar:

        # Check LANGSMITH_TRACING
        if hasattr(st.secrets, 'LANGSMITH') and hasattr(st.secrets.LANGSMITH, 'LANGSMITH_TRACING'):
            os.environ["LANGSMITH_TRACING"] = st.secrets.LANGSMITH.LANGSMITH_TRACING
        else:
            user_input = st.text_input("LangSmith Tracing (true/false)", value="true")
            if user_input:
                os.environ["LANGSMITH_TRACING"] = user_input

        # Check LANGSMITH_PROJECT
        if hasattr(st.secrets, 'LANGSMITH') and hasattr(st.secrets.LANGSMITH, 'LANGSMITH_PROJECT'):
            os.environ["LANGSMITH_PROJECT"] = st.secrets.LANGSMITH.LANGSMITH_PROJECT
        else:
            user_input = st.text_input("LangSmith Project Name", value="FreeStream")
            if user_input:
                os.environ["LANGSMITH_PROJECT"] = user_input

        # Check LANGSMITH_ENDPOINT
        if hasattr(st.secrets, 'LANGSMITH') and hasattr(st.secrets.LANGSMITH, 'LANGSMITH_ENDPOINT'):
            os.environ["LANGSMITH_ENDPOINT"] = st.secrets.LANGSMITH.LANGSMITH_ENDPOINT
        else:
            user_input = st.text_input("LangSmith Endpoint", type="password")
            if user_input:
                os.environ["LANGSMITH_ENDPOINT"] = user_input

        # Check LANGSMITH_API_KEY
        if hasattr(st.secrets, 'LANGSMITH') and hasattr(st.secrets.LANGSMITH, 'LANGSMITH_API_KEY'):
            os.environ["LANGSMITH_API_KEY"] = st.secrets.LANGSMITH.LANGSMITH_API_KEY
        else:
            user_input = st.text_input("LangSmith API Key", type="password")
            if user_input:
                os.environ["LANGSMITH_API_KEY"] = user_input

        # Check DEEPSEEK_API_KEY
        if hasattr(st.secrets, 'DEEPSEEK') and hasattr(st.secrets.DEEPSEEK, 'DEEPSEEK_API_KEY'):
            os.environ["DEEPSEEK_API_KEY"] = st.secrets.DEEPSEEK.DEEPSEEK_API_KEY
        else:
            user_input = st.text_input("DeepSeek API Key", type="password")
            if user_input:
                os.environ["DEEPSEEK_API_KEY"] = user_input

        # Check SERPAPI_KEY
        if hasattr(st.secrets, 'SERPAPI') and hasattr(st.secrets.SERPAPI, 'SERPAPI_KEY'):
            os.environ["SERPAPI_KEY"] = st.secrets.SERPAPI.SERPAPI_KEY
        else:
            user_input = st.text_input("Serp API Key", type="password")
            if user_input:
                os.environ["SERPAPI_KEY"] = user_input

    # # Display current environment status for debugging
    # st.write("Environment variables set:")
    # for key in ["LANGSMITH_TRACING", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT", 
    #             "LANGSMITH_API_KEY", "DEEPSEEK_API_KEY", "SERPAPI_KEY"]:
    #     if key in os.environ:
    #         # Mask sensitive keys in display
    #         display_value = '*' * len(os.environ[key]) if 'KEY' in key or 'SECRET' in key else os.environ[key]
    #         st.write(f"- {key}: {display_value}")
            
# st.button(label="load env", on_click=env_loader)