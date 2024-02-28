import streamlit as st

st.set_page_config(
    page_title="FreeStream: Unlimited Access to AI Tools",
    page_icon="🏡"
)

st.title(":rainbow[FreeStream]")
st.header(":green[_Unlimited Access to AI Tools_]", divider="red")
# Project Overview
st.subheader("What is FreeStream?")
st.write(
    """
    FreeStream is a project I'm working on to make AI tools more accessible. It's not just about having a chatbot; it's about exploring how AI can help us in our daily lives, today and in the future. Here's what you can do with FreeStream:

    *   **Explore AI Tools:** Dive into a wide range of AI tools, from chatbots to document analysis, to discover what's possible.
    *   **Educate Yourself:** Engage in interactive experiences designed to deepen your understanding of AI and its capabilities.
    *   **Solve Real-World Problems:** Utilize AI tools to simplify and enhance your daily tasks, showcasing the power of AI in action.
    """
)

st.subheader("What tools are currently available?")
st.write(
    """
    *   **RAGbot**: An AI chatbot designed to answer questions about documents.
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
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
height: 55px;
width: 100%;
background-color: white;
color: green;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/daemon-carino/" target="_blank">Daethyra</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)