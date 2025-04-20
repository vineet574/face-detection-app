import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.title("Face Detection App")

option = st.radio("Select Image Source:", ("Upload Image", "Use Webcam"))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img, len(faces)

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_np = np.array(img)
        result_img, face_count = detect_faces(img_np)
        st.image(result_img, caption=f"Detected {face_count} face(s)", channels="BGR")
else:
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame, face_count = detect_faces(frame)
        FRAME_WINDOW.image(result_frame, channels="BGR")
    cap.release()
