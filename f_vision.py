import cv2
import numpy as np
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import urllib.request

# 1. GEMINI SETUP 
genai.configure(api_key="AIzaSyDzELapQequcxI9y8RvJ8VO_AFUz_UnmUA") 
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Face Detector Auto-Download
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print("Face detector file missing! Auto-downloading from the internet...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)
    print("✅ Download Complete!")

# 3. LOCAL AI BRAIN LOAD
print("Loading AI Brain... Please wait!")
model = load_model('final_emotion_model.h5')
face_classifier = cv2.CascadeClassifier(cascade_path) 
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
print("Camera is on! Show your face on the screen. Press 'q' to close.")

def gemini_analyze(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    content = [
        "Act as a safety assistant. Analyze this image. The user is feeling fear. Is there any real danger or emergency visible in the background? Answer in 1 short English sentence.",
        {"mime_type": "image/jpeg", "data": encoded_image.tobytes()}
    ]
    try:
        response = gemini_model.generate_content(content)
        return response.text
    except Exception as e:
        return "Error connecting to Gemini."

gemini_cooling_down = False
cooldown_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)[0]
            fear_probability = prediction[2] * 100
            label = emotion_labels[np.argmax(prediction)]

            if fear_probability > 15.0:
                label = 'Fear'
                color = (0, 0, 255) 

                if not gemini_cooling_down:
                    print("\n🚨 Fear Detected! Sending image to Gemini...")
                    cv2.putText(frame, "ANALYZING EMERGENCY...", (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('NAPSS 3.0 Hybrid AI (Press Q to exit)', frame)
                    cv2.waitKey(1)

                    analysis = gemini_analyze(frame)
                    print(f"🤖 Gemini Says: {analysis}")
                    
                    gemini_cooling_down = True
                    cooldown_counter = 50 
            else:
                color = (0, 255, 0)

            if gemini_cooling_down:
                cooldown_counter -= 1
                if cooldown_counter <= 0:
                    gemini_cooling_down = False

            display_text = f"{label} (Fear: {int(fear_probability)}%)"
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('NAPSS 3.0 Hybrid AI (Press Q to exit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()