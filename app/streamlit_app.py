import streamlit as st
from models.utils import preprocess_image
from main import generate_caption

st.title("Image Captioning App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")
    caption = generate_caption(uploaded_file)
    st.write(f"Generated Caption: {caption}")