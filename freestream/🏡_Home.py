import streamlit as st

from freestream import footer

st.set_page_config(page_title="FreeStream: Unlimited Access to AI Tools", page_icon="🏡")

st.title("FreeStream")
st.header(":green[_Unlimited Access to AI Tools_]", divider="red")
# Project Overview
st.subheader(":blue[What is FreeStream?]")

st.write(
    """
    AI tools often seem complex or even intimidating, but FreeStream aims to change that. This project is about making AI accessible and understandable, showing how it can solve real-world problems in your daily life.
    """
)
st.divider()
st.subheader("What tools are currently available?")
st.write(
    """
    ### :blue[RAGbot]:
    
    :orange[*FreeStream's RAGbot can answer your questions directly from the documents you provide.*]
    
    This system empowers you to upload PDFs, Word documents, or plain text files. You can then pose specific questions directly related to the content of your documents.  Inspired by AlphaCodium's flow engineering techniques, it works as follows:

    1) Documentation Ingestion:  A long-context LLM carefully processes your uploaded documentation.

    2) Question Answering:  The system meticulously answers your question, drawing knowledge exclusively from the provided documents.

    3) Context Relevance Validation:  To safeguard against errors, the system only generates a response if the retrieved context is sufficiently relevant to aide the AI's response.
    
    ### :blue[Image Upscaler]:
    
    :orange[A proof of concept page, the *Image Upscaler* upscales images by 2.0x.]
    
    While the plan to use [Real-ESRGAN](), is in progress, [Swin2SR](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64) is used as a makeshift image upscaler. Due to the FreeStream's current environmental resource limitations, it cannot process images with a width or height greater than 300 pixels.
    """
)


st.markdown(
    """
    #### References
    
    * **[Run This App Locally](https://github.com/Daethyra/FreeStream/blob/streamlit/README.md#installation)**
    * **[Privacy Policy](https://github.com/Daethyra/FreeStream/blob/streamlit/README.md#privacy-policy)**
    * **[GitHub Repository](https://github.com/Daethyra/FreeStream)**    
    """
)

st.divider()

# Show footer
st.markdown(footer, unsafe_allow_html=True)
