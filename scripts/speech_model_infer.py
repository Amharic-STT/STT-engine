from librosa.core import audio
import streamlit as st
import os
import sys

from tokenizer import Tokenizer
import helper as helper
from model_helper import predict, build_model2
from model2 import CNN_net, BidirectionalRNN2, preprocessin_model
import librosa
import tensorflow as tf


# sys.path.append(os.path.abspath(os.path.join('./scripts')))

class Speech_Model_Infer:

    def __init__(self) -> None:

        self.sample_rate = 8000
        self.fft_size = 512
        self.frame_step = 256
        self.n_mels = 128
        self.output_dim = 223
        self.batch_size = 32

        self.preprocess_model = preprocessin_model(
            self.sample_rate, self.fft_size, self.frame_step, self.n_mels)

        self.cnn_model, self.cnn_shape = CNN_net(self.n_mels)
        self.BI_RNN_2 = BidirectionalRNN2(
            1024, batch_size=self.batch_size, output_dim=self.output_dim)

        self.cnn_bi_rnn_model = build_model2(
            self.output_dim, self.cnn_model, self.BI_RNN_2, self.preprocess_model)
        self.cnn_bi_rnn_model.load_weights("../models/cnn-bi-rnn.h5")

        self.int_to_char = helper.read_obj("../int_to_char.pkl")
        self.char_to_int = helper.read_obj("../char_to_int.pkl")

        self.tokenizer = Tokenizer(None)

    def predict(self, audio_file):

        wav, _ = librosa.load(audio_file, sr=self.sample_rate)
        pred, _ = predict(self.cnn_bi_rnn_model, wav,
                          self.tokenizer, self.int_to_char, None)
        return pred
    


if __name__ == "__main__":
    audio_obj = helper.read_obj("../data/test_audio_dict.pkl")
    trans_obj = helper.read_obj("../data/test_translation_dict.pkl")

    audios = []
    translations = []

    for label in audio_obj:
        audios.append(audio_obj[label][0])
        translations.append(trans_obj[label])

    SI = Speech_Model_Infer()
    tokenizer = SI.tokenizer
    cnn_bi_rnn_model = SI.cnn_bi_rnn_model
    int_to_char = SI.int_to_char
    
    for i in range(10):
      
      actual_translation = translations[i]
      sample_test_audio = audios[i]

      predicted, error = predict(cnn_bi_rnn_model, sample_test_audio,
                                tokenizer, int_to_char, actual=actual_translation)

      print("actual", actual_translation)
      print("predicted", predicted)
      print(f"WER: {error:.2f}")

      print()


#     translations.append(translation_obj[label])
#     print(SI.predict("./test2.wav"))
