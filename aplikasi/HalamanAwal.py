import streamlit as st
from  PIL import Image, ImageEnhance


st.title('Deteksi Masker')


uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns(3)

    with col2:
        st.markdown('<p style="text-align: center;">Gambar</p>',unsafe_allow_html=True)
        st.image(image,width=300)  
