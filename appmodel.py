import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# Function to load the trained models
def load_models():
    vae_model = tf.saved_model.load('vae_model_saved_model_format')  # Load the VAE model
    cnn_model = tf.saved_model.load('cnn_model_saved_model_format')
    #cnn_model = tf.keras.models.load_model('facial_expression_output.h5')# Load the CNN model
    return vae_model, cnn_model

vae_model, cnn_model = load_models()

def generate_output(input_image, model_type):
    if model_type == 'VAE':
        input_image_processed = np.array(input_image.resize((128, 128))) / 255.0
        input_image_processed = np.expand_dims(input_image_processed, axis=0)
        input_tensor = tf.convert_to_tensor(input_image_processed, dtype=tf.float32)
        output = vae_model.signatures['serving_default'](inputs=input_tensor)
        output_image = output['output_0'][0].numpy()  # Ensure output is a single image array
    elif model_type == 'CNN':
        input_image_processed = np.array(input_image.resize((48, 48))) / 255.0
        input_image_processed = np.expand_dims(input_image_processed, axis=0)
        input_tensor = tf.convert_to_tensor(input_image_processed, dtype=tf.float32)
        output = cnn_model.signatures['serving_default'](inputs=input_tensor)
        output_image = output['output_0'][0].numpy()

    # Check if output_image has a channel dimension and adjust if necessary
    if output_image.ndim == 2:
        output_image = np.stack((output_image,)*3, axis=-1)  # Convert grayscale to RGB
    elif output_image.ndim == 4:
        output_image = output_image[0]  # Select the first image if batch dimension is included

    return output_image

# Example usage in Streamlit
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='Uploaded Image', use_column_width=True)
    
    model_type = st.radio("Choose a model type:", ('VAE', 'CNN'))
    
    if st.button('Generate Output'):
        with st.spinner('Generating Output...'):
            output_image = generate_output(input_image, model_type)
            if output_image.shape[-1] == 1:  # Check if the last dimension is 1 (grayscale)
                output_image = np.squeeze(output_image, axis=-1)  # Remove the last dimension for grayscale
            st.image(output_image, caption='Generated Output', use_column_width=True)
            st.success('Output generated successfully!')


# # Streamlit UI
# st.title('Model Deployment: VAE and CNN')

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"], accept_multiple_files=False)

# if uploaded_file is not None:
#     input_image = Image.open(uploaded_file)
#     st.image(input_image, caption='Uploaded Image', use_column_width=True)
    
#     model_type = st.radio("Choose a model type:", ('VAE', 'CNN'))
    
#     if st.button('Generate Output'):
#         with st.spinner('Generating Output...'):
#             output_image = generate_output(input_image, model_type)
#             st.image(output_image, caption='Generated Output', use_column_width=True)
#             st.success('Output generated successfully!')

# def print_model_signatures(model):
#     for key, sig in model.signatures.items():
#         print(f"Signature: {key}")
#         print("Inputs:")
#         for input_name, input_tensor in sig.inputs.items():
#             print(f"  {input_name}: {input_tensor.shape}, {input_tensor.dtype}")
#         print("Outputs:")
#         for output_name, output_tensor in sig.outputs.items():
#             print(f"  {output_name}: {output_tensor.shape}, {output_tensor.dtype}")

def print_model_signatures(model):
    for key, sig in model.signatures.items():
        print(f"Signature: {key}")
        print("Inputs:")
        for input_tensor in sig.inputs:
            print(f"  {input_tensor.name}: {input_tensor.shape}, {input_tensor.dtype}")
        print("Outputs:")
        for output_tensor in sig.outputs:
            print(f"  {output_tensor.name}: {output_tensor.shape}, {output_tensor.dtype}")

# Call this function after loading your models
# For example: print_model_signatures(your_model)


# Call this function after loading your models
print_model_signatures(cnn_model)

