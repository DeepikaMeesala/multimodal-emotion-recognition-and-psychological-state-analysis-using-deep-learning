import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

st.title("Emotion Recognition App")

model = load_model("emotion_model.h5")

emotion_labels = [
"Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"
]

face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(48,48))
        face = face/255.0
        face = np.reshape(face,(1,48,48,1))

        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,emotion,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,0),2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)
