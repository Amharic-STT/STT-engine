import sys

from dataset_loader import load_audio_files, load_transcripts, load_spectrograms_with_transcripts, load_spectrograms_with_transcripts_in_batches
from resize_and_augment import resize_audios_mono, augment_audio, equalize_transcript_dimension
from FeatureExtraction import FeatureExtraction
from transcript_encoder import fit_label_encoder, encode_transcripts, decode_predicted
from models import model_1, model_2, model_3, model_4
from new_model import my_model
from jiwer import wer

import librosa   #for audio processing
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import warnings
import mlflow
import mlflow.keras
import os
import logging
len(os.listdir('./data/train/wav/'))

sample_rate = 44100

audio_files, maximum_length = load_audio_files('./data/train/wav/', sample_rate, True)
logging.info('loaded audio files')

print("The longest audio is", maximum_length/sample_rate, 'seconds long')
print("max length", maximum_length)

demo_audio = list(audio_files.keys())[0]

transcripts = load_transcripts("../data/train/trsTrain.txt")
logging.info('loaded transcripts')

audio_files = resize_audios_mono(audio_files, 440295)
print("resized shape", audio_files[demo_audio].shape)

audio_files = augment_audio(audio_files, sample_rate)
print("augmented shape", audio_files[demo_audio].shape)

char_encoder = fit_label_encoder(transcripts)
transcripts_encoded = encode_transcripts(transcripts, char_encoder)
enc_aug_transcripts = equalize_transcript_dimension(audio_files, transcripts_encoded, 200)

print('model summary')
#model = my_model(char_encoder, maximum_length)
import tensorflow as tf
from new_model import LogMelgramLayer, CTCLayer
model = tf.keras.models.load_model('../models/new_model_v1_2000.h5', 
                                    custom_objects = {
                                        'LogMelgramLayer': LogMelgramLayer ,
                                        'CTCLayer': CTCLayer}
                                    )
print(model.summary())

def load_data(audio_files, encoded_transcripts):
    X_train = []
    y_train = []
    for audio in audio_files:
        X_train.append(audio_files[audio])
        y_train.append(encoded_transcripts[audio])
    return np.array(X_train), np.array(y_train)

X_train, y_train = load_data(audio_files, enc_aug_transcripts)
print(X_train.shape, y_train.shape)
X_val, y_val = X_train[-10:], y_train[-10:]
X_test, y_test = X_train[:10], y_train[:10]#X_train[-20:-10], y_train[-20:-10]
X_train, y_train = X_train[:-20], y_train[:-20]

mlflow.set_tracking_uri('../')
mlflow.keras.autolog()

history = model.fit([X_train, y_train], 
                    validation_data = [X_val, y_val], 
                    batch_size = 25, epochs = 100)

model.save('../models/new_model_v1_10000.h5')

# with mlflow.start_run() as run:
#     mlflow.log_metric("ctc_loss", history.history['loss'][-1])

predicted = model.predict([X_test,y_test])
predicted_trans = decode_predicted(predicted, char_encoder)
real_trans = [''.join(char_encoder.inverse_transform(y)) for y in y_test]
for i in range(len(y_test)):
    print("Test", i)
    print("pridicted:",predicted_trans[i])
    print("actual:",real_trans[i])
    print("word error rate:", wer(real_trans[i], predicted_trans[i]))
    
WER = wer(predicted_trans[0], real_trans[0])

with open("metrics.json", 'w') as outfile:
    json.dump({"WER": WER}, outfile)

with open("prediction.txt", 'w') as outfile:
    outfile.write("Predicted: {}".format(predicted_trans[0]))
    outfile.write("Actual: {}".format(real_trans[0]))
