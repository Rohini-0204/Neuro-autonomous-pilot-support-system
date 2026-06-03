import cv2
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import scipy.io.wavfile as wav
import google.generativeai as genai
import speech_recognition as sr
import os
import time
import pygame 
import urllib.request
import threading 
import shutil
from datetime import datetime 

# --- STAGE 1: DYNAMIC DIRECTORY ANCHOR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UPDATE: Load from Hugging Face
from huggingface_hub import hf_hub_download
print("📥 Fetching model from Hugging Face...")
model_path = hf_hub_download(repo_id="Soniya2701/NAPSS", filename="final_emotion_model.h5")

FACE_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_eye.xml')

ALARM_AUDIO_PATH = os.path.join(BASE_DIR, 'danger_alarm.mp3')
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, 'audio_model.h5')

# --- NEW: Create a Blackbox folder for local reporting ---
REPORTS_DIR = os.path.join(BASE_DIR, 'EMERGENCY_REPORTS')
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    print(f"📁 Blackbox Directory Created: {REPORTS_DIR}")

if not os.path.exists(EYE_CASCADE_PATH):
    print("📥 Secondary telemetry file missing. Auto-downloading eye cascade...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    urllib.request.urlretrieve(url, EYE_CASCADE_PATH)

# --- STAGE 2: INITIALIZING LOGIC CORES & SOUND MIXER ---
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512) 

if os.path.exists(ALARM_AUDIO_PATH):
    pygame.mixer.music.load(ALARM_AUDIO_PATH)
    print("✅ Danger Alarm pre-loaded into RAM for zero-latency!")

genai.configure(api_key="AIzaSyDzELapQequcxI9y8RvJ8VO_AFUz_UnmUA")
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

print("🚀 NAPSS 3.0 Ultimate Guard (AI Decision Edition) Active...")

vision_model = load_model(model_path)
face_classifier = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_classifier = cv2.CascadeClassifier(EYE_CASCADE_PATH)

VISION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

try:
    audio_ai_model = load_model(AUDIO_MODEL_PATH)
    print("✅ Local Audio Emotion Model Loaded Successfully!")
except Exception as e:
    audio_ai_model = None

cooldown = False
is_unresponsive = False
system_running = True
last_report_saved = "" 
fear_prob = 0.0 

# --- AI DECISION TRACKING VARIABLES ---
ai_command_text = ""
ai_command_time = 0

# --- PURE OPENCV FATIGUE TRACKERS (NEW) ---
baseline_face_y = None
head_drop_start_time = None
is_yawning = False

# =====================================================================
# HUD UI DESIGN FUNCTIONS
# =====================================================================
def draw_hud_corners(frame, x, y, w, h, color, thickness=2, length=25):
    """Draws sci-fi style targeting brackets around the face"""
    cv2.line(frame, (x, y), (x + length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + length), color, thickness)
    cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y - length + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y - length + h), color, thickness)

# =====================================================================
# BLACKBOX LOCAL LOGGING FUNCTION
# =====================================================================
def save_blackbox_report(trigger_type, gemini_response, image_path, audio_path=None):
    global last_report_saved
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file_path = os.path.join(REPORTS_DIR, f"Incident_Report_{timestamp}.txt")
        
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write("====================================================\n")
            f.write("        NAPSS 3.0 OFFICIAL INCIDENT REPORT\n")
            f.write("====================================================\n")
            f.write(f"DATE & TIME    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"TRIGGER REASON : {trigger_type}\n")
            f.write("----------------------------------------------------\n")
            f.write("EVIDENCE FILES ATTACHED:\n")
            f.write(f"- Image: Snapshot_{timestamp}.jpg\n")
            if audio_path:
                f.write(f"- Audio: Voice_Cmd_{timestamp}.wav\n")
            else:
                f.write("- Audio: N/A (Visual Trigger)\n")
            f.write("----------------------------------------------------\n")
            f.write("MASTER CLOUD AI ANALYSIS:\n\n")
            f.write(gemini_response)
            f.write("\n====================================================\n")
            f.write("          SYSTEM AUTO-GENERATED LOG FILE\n")
            f.write("====================================================\n")
            
        final_img_path = os.path.join(REPORTS_DIR, f"Snapshot_{timestamp}.jpg")
        shutil.copy(image_path, final_img_path)
        
        if audio_path:
            final_aud_path = os.path.join(REPORTS_DIR, f"Voice_Cmd_{timestamp}.wav")
            shutil.copy(audio_path, final_aud_path)
            
        print(f"\n📂 [BLACKBOX] Incident Report Saved: {report_file_path}")
        last_report_saved = f"LAST REPORT SAVED: {timestamp}"
        
    except Exception as e:
        print(f"Error saving Blackbox report: {e}")

# =====================================================================
# MODULE 1: AUTO EMERGENCY SYSTEM (NO MIC, INSTANT ALARM)
# =====================================================================

def async_alarm_worker():
    try:
        if os.path.exists(ALARM_AUDIO_PATH):
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and system_running:
                time.sleep(0.1)
        else:
            print("\a") 
    except Exception as e:
        print(f"Siren execution exception: {e}")

def auto_emergency_handler(capture_frame, trigger_type):
    global cooldown, ai_command_text, ai_command_time
    try:
        print(f"\n🚨 [AUTO-SYSTEM] {trigger_type} INTERCEPT ACTIVATED!")
        temp_img = os.path.join(BASE_DIR, "evidence_img.jpg")
        cv2.imwrite(temp_img, capture_frame)
        
        img_file = genai.upload_file(path=temp_img)
        
        prompt = f"""
        Analyze this cockpit snapshot. 
        Trigger Reason: {trigger_type}
        
        Format your response exactly like this:
        - Transcription: N/A
        - Vocal Emotion & Stress: N/A (Visual Trigger Only)
        - Visual Status: [What is the physical condition in the image?]
        - System Verdict: [DANGER or SAFE]
        - AI Action Command: [Provide a short 3-6 word autonomous action command suitable for this emergency. Example: ENGAGING AUTO-PILOT AND REDUCING SPEED]
        """
        response = gemini_model.generate_content([prompt, img_file])
        
        for line in response.text.split('\n'):
            if "AI Action Command:" in line or "AI ACTION COMMAND:" in line.upper():
                ai_command_text = line.split(":")[-1].strip().upper()
                ai_command_time = time.time()
                break
                
        print("\n=================== 🚨 MASTER AI REPORT ===================")
        print(response.text)
        print("===================================================================\n")
        
        save_blackbox_report(trigger_type, response.text, temp_img)
        
    except Exception as e:
        print(f"Auto Emergency Error: {e}")
    finally:
        time.sleep(15)
        cooldown = False
        print("✅ Emergency System Restored. Monitoring Resumed.")

# =====================================================================
# MODULE 2: NEURO ASSISTANT SYSTEM (MIC ONLY, NO ALARM)
# =====================================================================

def neuro_assistant_handler(capture_frame):
    global cooldown, ai_command_text, ai_command_time
    try:
        print("\n🔔 [NEURO]: Hearing you! Speak for 5-7 seconds...")
        fs = 22050
        duration = 6.0
        
        try:
            cmd_chunk = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
        except sd.PortAudioError:
            cmd_chunk = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float32')
            sd.wait()
            if len(cmd_chunk.shape) > 1:
                cmd_chunk = np.mean(cmd_chunk, axis=1) 
        
        temp_aud = os.path.join(BASE_DIR, "neuro_cmd.wav")
        wav.write(temp_aud, fs, (cmd_chunk * 32767).astype(np.int16))
        
        print("⏳ Transcribing your exact words via Google API...")
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_aud) as cmd_source:
            cmd_data = recognizer.record(cmd_source)
            
        try:
            user_voice_command = recognizer.recognize_google(cmd_data).lower()
        except:
            user_voice_command = "(Audio unclear, but manual override logged)"
            
        print(f"\n🗣️ [USER COMMAND TRANSCRIBED]: \"{user_voice_command.upper()}\"")
        
        local_audio_status = "Not analyzed"
        if audio_ai_model is not None:
            try:
                import librosa
                audio_y, sr_lib = librosa.load(temp_aud, sr=22050)
                mfccs = librosa.feature.mfcc(y=audio_y, sr=sr_lib, n_mfcc=40)
                mfccs_scaled = np.mean(mfccs.T, axis=0)
                mfccs_scaled = np.expand_dims(mfccs_scaled, axis=0)
                
                audio_pred = audio_ai_model.predict(mfccs_scaled, verbose=0)
                confidence = np.max(audio_pred) * 100
                local_audio_status = f"Local Edge AI detected vocal anomaly with {confidence:.1f}% confidence."
            except:
                pass

        temp_img = os.path.join(BASE_DIR, "evidence_img.jpg")
        cv2.imwrite(temp_img, capture_frame)
        
        img_file = genai.upload_file(path=temp_img)
        aud_file = genai.upload_file(path=temp_aud)
        
        prompt = f"""
        Analyze this cockpit snapshot and audio. 
        Trigger Reason: MANUAL VOICE COMMAND
        
        Format your response exactly like this:
        - Transcription: The operator explicitly said: '{user_voice_command}'
        - Vocal Emotion & Stress: [Verify local AI result based on the voice]
        - Visual Status: [What is the physical condition in the image?]
        - System Verdict: [DANGER or SAFE]
        - AI Action Command: [Provide a short 3-6 word autonomous action command specifically based on what the pilot said. Example: INITIATING EMERGENCY BRAKING SYSTEM]
        """
        response = gemini_model.generate_content([prompt, img_file, aud_file])
        
        for line in response.text.split('\n'):
            if "AI Action Command:" in line or "AI ACTION COMMAND:" in line.upper():
                ai_command_text = line.split(":")[-1].strip().upper()
                ai_command_time = time.time() 
                break

        print("\n=================== 🧠 NEURO ASSISTANT REPORT ===================")
        print(response.text)
        print("===================================================================\n")
        
        save_blackbox_report("MANUAL VOICE COMMAND", response.text, temp_img, temp_aud)
        
    except Exception as e:
        print(f"Neuro Assistant Error: {e}")
    finally:
        cooldown = False
        print("✅ Neuro operations complete. System ready.")

# =====================================================================
# STAGE 3: LIVE TARGET TRACKING EXECUTION LOOP
# =====================================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fatal Alert: Camera device mapping failed.")
    exit()

eye_closed_start_time = None
print("✅ Active Monitoring System Environment: ONLINE")
print("👉 Press 'n' on keyboard to speak to Neuro.")
print("👉 Press 'q' or 'Esc' to exit safely.")

while system_running:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.25, 5)
    
    cv2.putText(frame, "NAPSS 3.0 SYSTEM HUD [ACTIVE]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    for (x, y, w_face, h_face) in faces:
        
        target_color = (0, 255, 0) if fear_prob < 18 else (0, 0, 255)
        draw_hud_corners(frame, x, y, w_face, h_face, target_color, thickness=2, length=25)
        
        # =================================================================
        # NEW: PURE OPENCV YAWN DETECTION (Blob Tracking)
        # =================================================================
        mouth_y_offset = int(h_face * 0.65)
        mouth_roi_gray = gray[y + mouth_y_offset : y + h_face, x : x + w_face]
        
        blur = cv2.GaussianBlur(mouth_roi_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_yawning = False
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mx, my, mw, mh = cv2.boundingRect(largest_contour)
            
            if mh > (h_face * 0.15) and mw > (w_face * 0.15):
                is_yawning = True
                cv2.rectangle(frame, (x + mx, y + mouth_y_offset + my), (x + mx + mw, y + mouth_y_offset + my + mh), (0, 255, 255), 2)
                cv2.putText(frame, "YAWN TRACED", (x + mx, y + mouth_y_offset + my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if is_yawning:
            cv2.putText(frame, "FATIGUE WARNING: YAWNING DETECTED!", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # =================================================================
        # NEW: PURE OPENCV HEAD DROP (MICRO-SLEEP) LOGIC
        # =================================================================
        center_y = y + (h_face // 2)
        
        if baseline_face_y is None:
            baseline_face_y = center_y 
        else:
            baseline_face_y = 0.95 * baseline_face_y + 0.05 * center_y

        is_head_dropped = False
        if (center_y - baseline_face_y) > 40:
            is_head_dropped = True

        if is_head_dropped:
            if head_drop_start_time is None:
                head_drop_start_time = time.time()
            head_drop_elapsed = time.time() - head_drop_start_time
            cv2.putText(frame, f"HEAD DROP DETECTED ({head_drop_elapsed:.1f}s)", (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            if head_drop_elapsed > 2.0 and not cooldown:
                cooldown = True
                trigger_reason = "MICRO-SLEEP (HEAD DROP)"
                threading.Thread(target=async_alarm_worker, daemon=True).start()
                threading.Thread(target=auto_emergency_handler, args=(frame.copy(), trigger_reason), daemon=True).start()
        else:
            head_drop_start_time = None

        # =================================================================
        # EYE & EMOTION TRACKING
        # =================================================================
        eye_roi_gray = gray[y : y + int(h_face * 0.55), x : x + w_face]
        eye_roi_color = frame[y : y + int(h_face * 0.55), x : x + w_face]
        eyes = eye_classifier.detectMultiScale(eye_roi_gray, 1.15, 6)
        
        if len(eyes) == 0:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            elapsed_time = time.time() - eye_closed_start_time
            cv2.putText(frame, f"GAZE: LOST ({elapsed_time:.1f}s)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if elapsed_time > 3.0 and not cooldown:
                is_unresponsive = True
        else:
            eye_closed_start_time = None
            is_unresponsive = False
            cv2.putText(frame, "GAZE: LOCKED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for (ex, ey, ew, eh) in eyes:
                center_x, center_y = ex + ew//2, ey + eh//2
                cv2.drawMarker(eye_roi_color, (center_x, center_y), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
        
        full_face_roi_gray = gray[y:y+h_face, x:x+w_face]
        roi_gray_resized = cv2.resize(full_face_roi_gray, (48, 48))
        
        if np.sum([roi_gray_resized]) != 0:
            roi_tensor = img_to_array(roi_gray_resized.astype('float') / 255.0)
            roi_tensor = np.expand_dims(roi_tensor, axis=0)

            prediction = vision_model.predict(roi_tensor, verbose=0)[0]
            max_index = int(np.argmax(prediction)) 
            dominant_emotion = VISION_LABELS[max_index]
            dominant_prob = prediction[max_index] * 100
            
            text_color = (0, 255, 255) if dominant_emotion != 'Fear' else (0, 0, 255)
            cv2.putText(frame, f"Emotion: {dominant_emotion} ({dominant_prob:.1f}%)", (x, y + h_face + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            fear_prob = min(prediction[2] * 100 * 2.5, 100)

            if (fear_prob > 25.0 or is_unresponsive) and not cooldown:
                cooldown = True
                trigger_reason = "COGNITIVE PANIC" if dominant_emotion == 'Fear' else "PILOT FREEZE"
                
                threading.Thread(target=async_alarm_worker, daemon=True).start()
                threading.Thread(target=auto_emergency_handler, args=(frame.copy(), trigger_reason), daemon=True).start()

    # --- HUD UI Elements ---
    bar_x, bar_y, bar_w, bar_h = 15, 120, 25, 250  
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
    fill_h = int((min(fear_prob, 100) / 100) * bar_h)
    bar_color = (0, 255, 0) if fear_prob < 25 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y + bar_h - fill_h), (bar_x + bar_w, bar_y + bar_h), bar_color, -1)
    
    cv2.putText(frame, "PANIC", (bar_x - 5, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{int(fear_prob)}%", (bar_x - 5, bar_y + bar_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 2)

    if cooldown:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0) 
        cv2.putText(frame, "!!! SYSTEM OVERRIDE IN PROGRESS !!!", (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    status_str = "SYSTEM STATUS: INTERCEPT" if (is_unresponsive or head_drop_start_time) else "SYSTEM STATUS: MONITORING"
    cv2.putText(frame, status_str, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if (is_unresponsive or head_drop_start_time) else (255, 255, 0), 2)
    
    mic_status = "[N] COMMAND: STANDBY" if not cooldown else "[N] COMMAND: OFFLINE"
    cv2.putText(frame, mic_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if not cooldown else (0, 165, 255), 2)

    if last_report_saved:
        cv2.putText(frame, last_report_saved, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # =====================================================================
    # AI AUTONOMOUS DECISION OVERLAY
    # =====================================================================
    if time.time() - ai_command_time < 15 and ai_command_text != "":
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 80), (frame.shape[1] - 50, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        cv2.putText(frame, f">> AI DECISION: {ai_command_text} <<", (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('NAPSS 3.0 Ultimate Guard', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: 
        print("\n⏹️ Manual Termination Sequence Initiated.")
        system_running = False
        break
        
    elif key == ord('n') and not cooldown:
        cooldown = True
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 255), -1) 
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        cv2.putText(frame, "!!! NEURO: HEARING YOU !!!", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('NAPSS 3.0 Ultimate Guard', frame)
        cv2.waitKey(1)
        
        threading.Thread(target=neuro_assistant_handler, args=(frame.copy(),), daemon=True).start()

system_running = False
pygame.mixer.quit() 
cap.release()
cv2.destroyAllWindows()
print("👋 Core System Safely Closed.")