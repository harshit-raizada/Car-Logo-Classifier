import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

@st.cache(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model('final_model.h5')
    return model

model = load_model()

st.title('Brand Logo Classification App')

uploaded_file = st.file_uploader("Choose an Image...", type = ["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    def preprocess_image(image, target_size=(224, 224)):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size)
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)
        return image

    processed_image = preprocess_image(image, target_size=(224, 224))

    predictions = model.predict(processed_image)
    st.write(predictions)

    class_names = ['hyundai', 'mercedes', 'skoda', 'toyota', 'volkswagen'] 
    stringed_predictions = [class_names[np.argmax(prediction)] for prediction in predictions]
    st.write(stringed_predictions)