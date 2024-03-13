import os

import streamlit as st
from pages.utils.styles import footer
from pages.utils.utility_funcs import download_model, upscaler_model_options

# Initialize page config
st.set_page_config(page_title="FreeStream: Real-ESRGAN", page_icon="🖼️")
st.title("🖼️Real-ESRGAN")
st.header(":green[_⚠️Under Construction⚠️_]", divider="red")
st.caption(":violet[_Placeholder._]")

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
# Body description
# Create a selectbox for model selection and download the selected model if not present
upscaler_model = st.selectbox("Select a model", list(upscaler_model_options.keys()))

if st.button("Download Model"):
    with st.container():
        download_model(upscaler_model, upscaler_model_options[upscaler_model])


st.divider()

# Create two columns with a single row to organize the UI
left_image, right_image = st.columns(2)
# Define a container for image containers
image_showcase = st.container()  # holds other containers

with image_showcase:
    try:
        if uploaded_files:
            # Show the uploaded image
            with left_image:
                st.image(uploaded_files)  # Latest uploaded image

            # Upscale and show the upscaled image
            # with right_image:
            #    upscaled_image = image_upscaler(uploaded_files)
            #    st.image(upscaled_image) # Latest uploaded image
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
