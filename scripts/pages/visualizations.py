''' This is the home page'''

# Libraries
import streamlit as st

def write():
    with st.spinner("Loading Home ..."):
        
        st.markdown("<p style='padding:30px;text-align:center; background-color:#3761B5;color:#FFFFFF;font-size:26px;border-radius:10px;'>Amharic speech recognition</p>", unsafe_allow_html=True)

        st.write(
            """
            Data visualizations
            
            """
        )
        
