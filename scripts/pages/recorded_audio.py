import streamlit as st
from scripts.pages.test_model import *
from pydub import AudioSegment

def write():
    st.markdown("<p style='padding:30px;text-align:center; background-color:#3761B5;color:#FFFFFF;font-size:26px;border-radius:10px;'>Amharic speech recognition</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select file from your directory")
    if uploaded_file is not None:
        
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        file_var = AudioSegment.from_wav(uploaded_file) 
        file_var.export('./data/wav/new.wav', format='wav')
        
        st.header("Transcribed text ") 
        with st.spinner("Processing....."):
            write1()
            
              
        st.balloons()  
           
