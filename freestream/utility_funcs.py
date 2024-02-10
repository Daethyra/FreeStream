# Define a callback function for when a model is selected
def set_llm():
    # Set the model in session state
    st.session_state.llm = model_names[selected_model]
    
    # Show an alert based on what model was selected
    if st.session_state.model_selector == model_names["ChatOpenAI GPT-3.5 Turbo"]:
        st.warning(body="Switched to ChatGPT 3.5-Turbo!", icon="⚠️")
    else:
        st.warning(body="Failed to change model! \nPlease contact the website builder.", icon="⚠️")