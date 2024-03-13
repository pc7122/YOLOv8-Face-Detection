import cv2
import streamlit as st
from ultralytics import YOLO

# Set up the YOLOv5 model
model = YOLO('./model/face_detector.pt')

# streamlit app
st.set_page_config(page_title='Face Detection - YOLOv8')

st.markdown('''
    <h1 style="text-align: center;">Face Detection - YOLOv8</h1>''', unsafe_allow_html=True)

frame = st.image('./cam_thumbnail.jpeg', use_column_width=True)

col1, col2 = st.columns(2)

with col1:
    start_button = st.button('Start', use_container_width=True, type='primary')

with col2:
    stop_button = st.button('Stop', use_container_width=True)

if start_button:
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)
        frame.image(results[0].plot())

        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()
