import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# Function to load the trained VAE model
def load_model():
    return tf.saved_model.load('vae_model_saved_model_format')  # Load the SavedModel format

vae_model = load_model()

# Function to preprocess input image and generate output from the VAE model
# def generate_output(input_image):
#     # Preprocess the input image (resize, normalize, etc.)
#     input_image = np.array(input_image.resize((128, 128))) / 255.0
#     input_image = np.expand_dims(input_image, axis=0)

#     # Generate output from the VAE model
#     #output_image = vae_model(input_image)
#     output_image = vae_model(input_image)# Assuming your VAE model directly takes input and generates output
#     return output_image[0]  # Return the generated output image

def generate_output(input_image):
    input_image = np.array(input_image.resize((128, 128))) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Assuming the model exposes a callable serving signature:
    output_image = vae_model.signatures['serving_default'](tf.convert_to_tensor(input_image))['output_0']

    return output_image[0].numpy()

# Streamlit UI
st.title('VAE Model Deployment')

#uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"], accept_multiple_files=False)

# if uploaded_file is not None:
#     with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     st.success("File successfully uploaded.")


if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='Uploaded Image', use_column_width=True)
    if st.button('Generate Output'):
        with st.spinner('Generating Output...'):
            output_image = generate_output(input_image)
            st.image(output_image, caption='Generated Output', use_column_width=True)
            st.success('Output generated successfully!')
