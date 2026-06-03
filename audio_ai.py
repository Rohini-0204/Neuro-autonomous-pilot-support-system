import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import scipy.io.wavfile as wav
import google.generativeai as genai
import os
import time

# ---------------------------------------------------------
# 1. GEMINI AUDIO SETUP
# ---------------------------------------------------------
genai.configure(api_key="AIzaSyDzELapQequcxI9y8RvJ8VO_AFUz_UnmUA") 
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite') 

print("🎧 Loading Audio Brain... Please wait!")

# 2. Local Audio Model Load Kora
try:
    model = load_model('audio_model.h5')
    print("✅ Local Audio Brain Loaded!")
except Exception as e:
    print("❌ Error: Could not find 'audio_model.h5'!")
    exit()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 3. Audio features ber korar function
def extract_features(audio_data, sample_rate):
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    features = np.expand_dims(mfccs, axis=0)
    features = np.expand_dims(features, axis=2)
    return features

# 4. Mic diye kotha record korar function (Time barano hoyeche 15 seconds e)
def record_audio(duration=15, fs=22050):
    print(f"\n🎤 Speak now! Listening for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float32')
    sd.wait() 
    print("🛑 Recording Stopped! Analyzing...")
    
    audio_data = np.squeeze(recording)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
        
    return audio_data, fs

# 5. Gemini Audio Analysis Function (TRANSCRIBE & ANALYZE IN ENGLISH)
def ask_gemini_audio(filename):
    print("🤖 Master AI is processing the audio...")
    try:
        audio_file = genai.upload_file(path=filename)
        
        print("⏳ Uploading to Google Servers, please wait 2 seconds...")
        time.sleep(2) 
        
        prompt = """
        You are an expert safety assistant. Listen to the uploaded audio. The user is speaking in either Bengali or English. 
        Please provide your response strictly in English, formatted exactly like this:

        Transcription: [Write down what the user said in English translation or exact English words]
        Analysis: [In 1 short sentence, explain if they are in danger, scared, confused, or safe]
        """
        
        response = gemini_model.generate_content([prompt, audio_file])
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
print("\n--- NAPSS 3.0 SMART AUDIO AI ---")
print("Microphone is ready. Press Enter to test!")

while True:
    user_input = input("\n[Press 'Enter' to start testing, or type 'q' to quit]: ")
    if user_input.lower() == 'q':
        break

    # Record audio (15 seconds)
    audio_data, sr = record_audio(duration=15)
    
    try:
        features = extract_features(audio_data, sr)
        prediction = model.predict(features, verbose=0)[0]
        fear_prob = prediction[2] * 100
        sad_prob = prediction[5] * 100
        
        max_index = np.argmax(prediction)
        label = emotion_labels[max_index]

        print(f"\n🧠 Local AI Thinks: {label} (Fear: {fear_prob:.1f}%, Sad: {sad_prob:.1f}%)")
        
        if fear_prob > 10.0 or label == 'Sad':
            print("⚠️ Suspicious Tone Detected! Sending audio to Gemini for deep analysis...")
            
            temp_filename = "temp_record.wav"
            wav.write(temp_filename, sr, (audio_data * 32767).astype(np.int16))
            
            gemini_result = ask_gemini_audio(temp_filename)
            print("\n" + "="*50)
            print("🚨 MASTER AI REPORT:")
            print(gemini_result)
            print("="*50 + "\n")
            
    except Exception as e:
        print(f"\n⚠️ Error: {e}")