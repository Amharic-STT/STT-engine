import streamlit as st
import streamlit as st
import sounddevice as sd
import wavio
import librosa.display
from scripts.pages.test_model import *
def record(duration=10, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording

def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None
def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes    

def write():
    st.markdown("<p style='padding:30px;text-align:center; background-color:#3761B5;color:#FFFFFF;font-size:26px;border-radius:10px;'>Amharic speech recognition</p>", unsafe_allow_html=True)

    st.header("Record your own voice")

    filename = st.text_input("Choose a filename: ")

    if st.button(f"Click to Record"):
        if filename == "":
            st.warning("Choose a filename.")
        else:
            record_state = st.text("Recording...")
            duration = 5  # seconds
            fs = 48000
            myrecording = record(duration, fs)
            record_state.text(f"Saving sample as {filename}.mp3")

            path_myrecording = f"data/wav/{filename}.mp3"

            save_record(path_myrecording, myrecording, fs)
            record_state.text(f"Done! Saved sample as {filename}.mp3")
            st.audio(read_audio(path_myrecording))
            st.header("Transcribed text ") 
            with st.spinner("Processing....."):
                 write1()
            
              
            st.balloons()  
           
          
            


    


            
