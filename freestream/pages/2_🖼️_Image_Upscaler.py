import streamlit as st
from pages.utils.utility_funcs import image_upscaler

st.set_page_config(page_title="FreeStream: Image Upscaler", page_icon="🖼️")
st.title("🖼️Image Upscaler")
st.header(":green[_Header_]", divider="red")
st.caption(":violet[_Caption_]")
st.sidebar.subheader("__User Panel__")

uploaded_files = st.sidebar.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    help="Click the button and upload a *single* image.",
    accept_multiple_files=True,
)

# Create two columns with a single row to organize the UI
images_row = st.collumns(2)
# Define a container for image containers
image_showcase = st.container() # holds other containers
left_image = st.container() # holds uploaded image
right_image = st.container() # holds upscaled image

#for col in images_row:
with image_showcase:
    if uploaded_files:
        
        # Show the uploaded image
        with left_image:
            st.image(uploaded_files[0])
        
        # Upscale and show the upscaled image
        with right_image:
            st.image(image_upscaler(uploaded_files[0]))