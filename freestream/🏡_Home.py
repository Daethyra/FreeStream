import streamlit as st

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
    
    It works by allowing you to upload PDFs, Word documents, or plain text files, and then ask specific questions based on the information in your documents. The RAGbot uses a method called Corrective Retrieval Augmented Generation (CRAG), which involves retrieving documents, grading them for relevance, and generating answers if at least one document is relevant. If all documents are ambiguous or incorrect, it retrieves from an external source and uses that as context for answer generation. This process ensures a neat workflow where retrieval is done similarly to basic RAG, but with an added step of reasoning about the documents to ensure accurate and helpful responses.
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

# Create a footer using community suggestion:
# https://discuss.streamlit.io/t/streamlit-footer/12181
footer = """<style>
a:link , a:visited{
    color: #ffffff; /* White */
    background-color: transparent;
    text-decoration: underline;
}

a:hover, a:active {
    color: #cccccc; /* Light grey for hover */
    background-color: transparent;
    text-decoration: underline;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    height: 55px;
    width: 100%;
    background-color: #343a40; /* Dark grey */
    color: #ffffff; /* White for text */
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a href="https://www.linkedin.com/in/daemon-carino/" target="_blank">Daethyra</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
