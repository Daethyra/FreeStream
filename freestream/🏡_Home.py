import streamlit as st

st.set_page_config(page_title="FreeStream: Unlimited Access to AI Tools", page_icon="🏡")

st.title("FreeStream")
st.header(":green[_Unlimited Access to AI Tools_]", divider="red")
# Project Overview
st.subheader("What is FreeStream?")

st.write(
    """
    AI tools often seem complex or even intimidating, but FreeStream aims to change that. This project is about making AI accessible and understandable, showing how it can solve real-world problems in your daily life.
    """
)
st.divider()
st.subheader("What tools are currently available?")
st.write(
    """
    FreeStream's RAGbot can answer your questions directly from the documents you provide. Here's how it works:

    * **Upload Your Files:**  Share PDFs, Word documents, or plain text files. 
    * **Ask Your Question:**  Get specific answers based on the information in your documents.
    * **Harness Advanced AI:** RAGbot uses reflective retrieval for accurate results.
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
