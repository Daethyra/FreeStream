import os
import streamlit as st
from pages.utils.utility_funcs import image_upscaler

# Set environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Initialize page config
st.set_page_config(page_title="FreeStream: Image Upscaler", page_icon="🖼️")
st.title("🖼️Image Upscaler")
st.header(":green[_Header_]", divider="red")
st.caption(":violet[_Caption_]")
st.sidebar.subheader("__User Panel__")

uploaded_files = st.sidebar.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    help="Click the button and upload a *single* image.",
)

# Create two columns with a single row to organize the UI
left_image, right_image = st.columns(2)
# Define a container for image containers
image_showcase = st.container() # holds other containers

with image_showcase: # add a try/except block
    if uploaded_files:
        
        # Show the uploaded image
        with left_image:
            st.image(uploaded_files) # Latest uploaded image
        
        # Upscale and show the upscaled image
        with right_image:
            st.image(image_upscaler(uploaded_files)) # Latest uploaded image