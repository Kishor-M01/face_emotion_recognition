import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "emotion_model.keras"
IMG_SIZE   = 96
EMOTIONS   = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
COLORS     = [(0,0,255),(128,0,128),(255,165,0),(0,255,0),(200,200,200),(255,0,0),(0,255,255)]

model       = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess(face_img):
    img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img / 255.0, axis=0)

def predict_emotion(face_img):
    preds = model.predict(preprocess(face_img), verbose=0)[0]
    idx   = np.argmax(preds)
    return EMOTIONS[idx], preds[idx], COLORS[idx]

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(48, 48))

    for (x, y, w, h) in faces:
        face_roi          = frame[y:y+h, x:x+w]
        emotion, conf, color = predict_emotion(face_roi)
        label             = f"{emotion}: {conf*100:.1f}%"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    cv2.imshow("Face Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
