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
    height: 40px;
    width: 100%;
    background-color: #343a40; /* Dark grey */
    color: #ffffff; /* White for text */
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px;
    z-index: 1; /* Ensure footer is on top of images, but not above other elements */
}

.footer p {
    margin: 0;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a href="https://www.linkedin.com/in/daethyra-carino/" target="_blank">Daethyra</a>.</p>
</div>
"""

message_background_shading = """
<style>
/* Target the main chat message container - now semi-transparent */
div[data-testid="stChatMessage"] {
    background-color: rgba(26, 26, 26, 0.7) !important; /* Added alpha channel for transparency */
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    border-left: 4px solid rgba(76, 175, 80, 0.8); /* Semi-transparent green */
    backdrop-filter: blur(5px); /* Adds frosted glass effect */
}
"""