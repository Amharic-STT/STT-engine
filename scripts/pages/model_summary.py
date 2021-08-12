

# Libraries
import streamlit as st
import tensorflow as tf
from scripts.new_model import LogMelgramLayer, CTCLayer

def load_model():
    model = tf.keras.models.load_model('./models/new_model_v1_2000.h5', 
                                            custom_objects = {
                                                'LogMelgramLayer': LogMelgramLayer ,
                                                'CTCLayer': CTCLayer}
                                            )
    return model
    
def write():
    with st.spinner("Loading..."):
        
        st.markdown("<p style='padding:30px;text-align:center; background-color:#3761B5;color:#FFFFFF;font-size:26px;border-radius:10px;'>Amharic speech recognition</p>", unsafe_allow_html=True)

        st.header("Model summary")
        model = load_model()
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        st.write(short_model_summary)
