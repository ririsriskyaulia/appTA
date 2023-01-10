import streamlit as st
from  PIL import Image, ImageEnhance
import pickle
import numpy as np
from skimage.io import imread, imsave
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing

st.title('Deteksi Masker')

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gambar = Image.open(img_file_buffer)
    col1, col2, col3 = st.columns(3)
    with col2:
            st.markdown('<p style="text-align: center;">Gambar</p>',unsafe_allow_html=True)
            st.image(gambar,width=300)  
            if st.button('predict'):
                class_label = {0: 'Mask', 1: 'No Mask'}
                model=tf.keras.models.load_model('model (1).h5')
                test_image = gambar.resize((150,150))
                test_image = preprocessing.image.img_to_array(test_image)
                img = np.reshape(test_image,[1,150,150,3])
                img_arr = np.zeros((1,150,150,3))
                img_arr[0, :, :, :] = img / 255.
                y_pred=model.predict(img_arr)
                
                st.success(class_label[np.argmax(y_pred)])
