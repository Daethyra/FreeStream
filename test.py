import streamlit as st
import os
import requests

def download_model(model_name, file_url):
    """
    Downloads a specified model from a given URL if it's not already present in the local directory.

    Parameters:
    - model_name (str): The name of the model to download. This is used to construct the local file path.
    - file_url (str): The URL from which to download the model file.

    Returns:
    - None
    """
    # Define the directory where models will be stored.
    models_dir = 'models'
    
    # Check if the models directory exists. If it doesn't, create it.
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Construct the local file path for the model.
    model_path = os.path.join(models_dir, f'{model_name}.pth')
    
    # Check if the model file already exists in the local directory.
    if not os.path.isfile(model_path):
        # Use st.spinner to show a spinner while the download is in progress.
        with st.spinner(f"Downloading {model_name}..."):
            # Send a GET request to the provided file URL to download the model file.
            response = requests.get(file_url)
            
            # Open the local file path in write-binary mode ('wb').
            with open(model_path, 'wb') as f:
                # Write the content of the response to the local file.
                f.write(response.content)
        
        # Use st.success to show a success message once the download is complete.
        st.success(f"{model_name} downloaded successfully!")
        
        # Optionally, use st.toast to display a toast notification for additional feedback.
        st.toast(f"{model_name} has been downloaded and is ready to use.", duration=3)
    else:
        # If the model file already exists, inform the user.
        st.info(f"{model_name} is already downloaded and ready to use.")

# Define model options and their corresponding file URLs
model_options = {
    'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'RealESRNet_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
    'RealESRGAN_x4plus_anime_6B': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
    'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    'realesr-animevideov3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
    'realesr-general-x4v3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
}

# Create a selectbox for model selection and download the selected model if not present
selected_model = st.selectbox("Select a model", list(model_options.keys()))

if st.button("Download Model"):

    with st.container():
        download_model(selected_model, model_options[selected_model])
