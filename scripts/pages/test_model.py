from scripts.dataset_loader import *
from  scripts.resize_and_augment import *
from  scripts.transcript_encoder import  *
import numpy as np
import os, shutil
import logging
import pickle
import tensorflow as tf
from scripts.new_model import LogMelgramLayer, CTCLayer
import streamlit as st



def perform_predictions(path):
    sample_rate = 44100
    audio_files, maximum_length = load_audio_files(path, sample_rate, True)
    logging.info('loaded audio files')
    audio_files = resize_audios_mono(audio_files, 440295)
    enc = open('./models/encoder.pkl', 'rb')
    char_encoder = pickle.load(enc)
    print('model summary')

    def load_model():
        model = tf.keras.models.load_model('./models/new_model_v1_2000.h5', 
                                            custom_objects = {
                                                'LogMelgramLayer': LogMelgramLayer ,
                                                'CTCLayer': CTCLayer}
                                            )
        return model
    model = load_model()
    #print(model.summary())

    def load_data(audio_files):
        X_train = []
        y_train = []
        for audio in audio_files:
            X_train.append(audio_files[audio])
            
        return np.array(X_train), np.array(y_train)

    X_test,y_test = load_data(audio_files)
   
    if len(y_test) == 0:
        y_test = np.array([[0]*70])
    predicted = model.predict([X_test,[y_test]])
    predicted_trans = decode_predicted(predicted, char_encoder)

    return predicted_trans
def write1():
    st.write(perform_predictions('./data/wav/')[-1])
    folder = './data/wav'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



