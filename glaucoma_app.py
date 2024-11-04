import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set Streamlit option to suppress deprecation warning
# st.set_option('deprecation.showfileUploaderEncoding', False)

# Function to preprocess image and make predictions
def import_and_predict(image_data, model):
    # Resize image to the required input size of the model
    image = ImageOps.fit(image_data, (100, 100), Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, channels='RGB')
    
    # Normalize the image
    image = image.astype(np.float32) / 255.0
    img_reshape = np.expand_dims(image, axis=0)
    
    # Prediction
    prediction = model.predict(img_reshape)
    return prediction

# Load the model (ensure 'my_model2.h5' is compatible with the current TensorFlow version)
model = tf.keras.models.load_model('my_model2.h5')

# Streamlit interface
st.write("""
         # ***Glaucoma Detector***
         This is a simple image classification web app to predict glaucoma through fundus image of the eye.
         """)

file = st.file_uploader("Please upload an image (jpg) file", type=["jpg"])

# Check if a file has been uploaded
if file is None:
    st.text("You haven't uploaded an image file.")
else:
    # Open image
    imageI = Image.open(file)
    
    # Make predictions
    prediction = import_and_predict(imageI, model)
    pred = prediction[0][0]
    
    # Display results
    if pred > 0.5:
        st.write("## **Prediction:** Your eye appears healthy. Great!")
        st.balloons()
    else:
        st.write("## **Prediction:** You may be affected by Glaucoma. Please consult an ophthalmologist.")
