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
