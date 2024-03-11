import os

import streamlit as st
from pages.utils.utility_funcs import image_upscaler
from pages.utils.styles import footer

# Set expandable_segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize page config
st.set_page_config(page_title="FreeStream: Image Upscaler", page_icon="🖼️")
st.title("🖼️Image Upscaler")
st.header(":green[_⚠️Under Construction⚠️_]", divider="red")
st.caption(
    ":violet[_This page is still under construction. Processing speed and output quality will improve over time._]"
)

# Show footer
st.markdown(footer, unsafe_allow_html=True)

# Create the sidebar
st.sidebar.subheader("__User Panel__")
# Add a file-upload button
uploaded_files = st.sidebar.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    help="Click the button and upload a *single* image.",
    key="image_to_upscale",
)

st.divider()
# Create a body paragraph
st.markdown(
    """
    Although the page is still being worked on, you're encouraged to test out the current upscaler, [Swin2SR](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64).
    
    **Limitations**:
    
    * Images with a width *or* height greater than 128 will be downsampled by 3/20ths. This limitation will be removed once Real-ESRGAN is implemented.
    * The current upscaler problematically generates new content around the edge of the image, especially on the right side.
    
    As with all FreeStream pages, this one's purpose is merely to show you the possibilities without having to sign up or program anything manually. 
    Right now, having a page that "works" is an important step in that it lays down the foundation for a robust solution.
    """
)

st.divider()

# Create two columns with a single row to organize the UI
left_image, right_image = st.columns(2)
# Define a container for image containers
image_showcase = st.container()  # holds other containers

with image_showcase:  # add a try/except block
    if uploaded_files:
        # Show the uploaded image
        with left_image:
            st.image(uploaded_files)  # Latest uploaded image

        # Upscale and show the upscaled image
        with right_image:
            st.image(image_upscaler(uploaded_files))  # Latest uploaded image
