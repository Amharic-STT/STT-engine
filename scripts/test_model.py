from  dataset_loader import load_audio_files, load_transcripts, load_spectrograms_with_transcripts, load_spectrograms_with_transcripts_in_batches
from  resize_and_augment import resize_audios_mono, augment_audio, equalize_transcript_dimension
from  transcript_encoder import fit_label_encoder, encode_transcripts, decode_predicted
import numpy as np
import os
import logging
import pickle
import tensorflow as tf
from new_model import LogMelgramLayer, CTCLayer
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
    print(model.summary())

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

z=perform_predictions('./data/wav/')
pribt(type(z))

