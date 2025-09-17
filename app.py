import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from deep_translator import GoogleTranslator
from tensorflow.keras.models import load_model
import os

def convert_2_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def binarization(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return thresh

def dilate(image, words=False):
    img = image.copy()
    m = 3
    n = m - 2
    itrs = 4
    if words:
        m = 6
        n = m
        itrs = 3
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))
    dilation = cv2.dilate(img, rect_kernel, iterations=itrs)
    return dilation

def find_rect(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rects.append([x, y, w, h])
    sorted_rects = sorted(rects, key=lambda x: x[0])
    return sorted_rects

# Assuming mapping_inverse is defined elsewhere
mapping_inverse = {i: chr(i + 65) for i in range(26)}  # Example mapping (A-Z), adjust as needed

def extract(image):
    model = load_model('CustomCnn_model.h5')
    if model is None:
        return None

    chars = []
    image_cpy = image.copy()
    bin_img = binarization(convert_2_gray(image_cpy))
    full_dil_img = dilate(bin_img, words=True)
    words = find_rect(full_dil_img)

    for word in words:
        x, y, w, h = word
        img = image_cpy[y:y+h, x:x+w]

        bin_img = binarization(convert_2_gray(img))
        dil_img = dilate(bin_img)
        char_parts = find_rect(dil_img)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

        for char in char_parts:
            x, y, w, h = char
            ch = img[y:y+h, x:x+w]

            empty_img = np.full((32, 32, 1), 255, dtype=np.uint8)
            x, y = 3, 3
            resized = cv2.resize(ch, (16, 22), interpolation=cv2.INTER_CUBIC)
            gray = convert_2_gray(resized)
            empty_img[y:y+22, x:x+16, 0] = gray.copy()
            gray = cv2.cvtColor(empty_img, cv2.COLOR_GRAY2RGB)
            gray = gray.astype(np.int32)

            predicted = mapping_inverse[np.argmax(model.predict(np.array([gray]), verbose=0))]
            chars.append(predicted)
        chars.append(' ')

    return ''.join(chars[:-1]).lower()

from deep_translator import GoogleTranslator

def translate_text(text, source_lang, target_lang='en'):
    return GoogleTranslator(source=source_lang, target=target_lang).translate(text)

def main():
    st.title("Image Text Extraction and Translation App")

    # Language selection dropdown
    language_mapping = {
        "French": "fr",
        "Spanish": "es",
        "Latin": "la",
    }
    selected_language = st.selectbox("Select source language:", list(language_mapping.keys()))

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=False, width=300)

        # Button for prediction (text extraction and translation)
        if st.button('Predict'):
            with st.spinner("Processing image..."):
                # Extract text from image
                extracted_text = extract(image.copy())

                if extracted_text:
                    # Translate the extracted text
                    translated_text = translate_text(extracted_text, language_mapping[selected_language])

                    # Display extracted and translated text
                    st.subheader("Extracted Text:")
                    st.write(extracted_text)

                    st.subheader("Translated Text (English):")
                    st.write(translated_text)

                    st.success("Processing complete!")
                else:
                    st.error("Failed to extract text from image")
if __name__ == "__main__":
    main()
