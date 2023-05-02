#import the packages 
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(layout="wide")
headerIMG = Image.open('CSI_header.jpg')
st.image(headerIMG)


@st.cache_resource(show_spinner=True)
class_labels = ['CSI','NonCSI']
def load_model():
  model=tf.keras.models.load_model('VGG19.h5')
  return model

model = tf.keras.models.load_model('VGG19.h5')


file = st.file_uploader("", type=["jpg", "png", "jpeg"])

st.set_option('deprecation.showfileUploaderEncoding', False)



def import_and_predict(image_data, model):
    
        size = (300,300)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img_reshape = image[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image,caption= "Image Uploaded", width = 400)
    predictions = import_and_predict(image, model)
    pred_label = class_labels[np.argmax(predictions[0])]
    score = tf.nn.softmax(predictions[0])
    st.write(pred_label)
    st.write("This image most likely belongs to {} with a {:.2f} percent confidence.".format(pred_label, 100 * np.max(score)))

