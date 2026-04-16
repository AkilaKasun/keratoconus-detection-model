import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load  saved model
model = tf.keras.models.load_model('keratoconus_model.h5')

st.title("Keratoconus Detection AI")
st.write("Upload a corneal topography map to check for Keratoconus.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Report', use_column_width=True)
    
    # Prepare image for AI
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    classes = ['Keratoconus', 'Normal', 'Suspect']
    result = classes[np.argmax(prediction)]
    
    st.subheader(f"Result: {result}")
    st.write(f"Confidence: {np.max(prediction)*100:.2f}%")