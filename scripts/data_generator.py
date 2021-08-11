from tokenizer import Tokenizer
import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,  translations, audios, batch_size=32, shuffle=True):
        self.audios = audios
        self.labels = translations
        self.batch_size = batch_size
        self.len = int(np.floor(len(self.labels) / self.batch_size))
        self.shuffle = shuffle
        self.on_epoch_end()

        self.tokenizer = Tokenizer(translations)
        self.int_to_char, self.char_to_int = self.tokenizer.build_dict()

        self.cur_index = 0

    def __len__(self):
        return self.len

    def encode_text(self, translations):
        encoded_trans = []

        for t in translations:
            encoded = self.tokenizer.encode(t, self.char_to_int)
            encoded_trans.append(encoded)

        return encoded_trans

    def get_max_len(self, items):
        maximum = 0
        for i in items:
            if len(i) > maximum:
                maximum = len(i)

        return maximum

    def __data_generation(self, batch_translations, batch_audios):

        self.cur_index = 0
        encoded_trans = self.encode_text(batch_translations)

        maximum_trans_len = self.get_max_len(encoded_trans)
        maximum_audio_len = self.get_max_len(batch_audios)

        encoded_trans_np = np.zeros(
            (len(encoded_trans), maximum_trans_len), dtype="int64")
        padded_audios_np = np.zeros(
            (len(batch_audios), maximum_audio_len), dtype="float32")

        label_length = np.zeros(padded_audios_np.shape[0], dtype="int64")
        input_length = np.zeros(encoded_trans_np.shape[0], dtype="int64")

        ind = 0
        for trans, audio in zip(encoded_trans, batch_audios):
            encoded_trans_np[ind, 0:len(trans)] = trans
            label_length[ind] = len(trans)

            padded_audio = np.pad(
                audio, (0, maximum_audio_len - len(audio)), mode='constant', constant_values=0)

            padded_audios_np[ind, ] = padded_audio
            input_length[ind] = len(audio)

            ind += 1

        outputs = {'ctc': np.zeros([self.batch_size])}
        inputs = {'the_input':   tf.convert_to_tensor(padded_audios_np),
                  'the_labels':   tf.convert_to_tensor(encoded_trans_np),
                  'input_length':   tf.convert_to_tensor(input_length),
                  'label_length':   tf.convert_to_tensor(label_length)
                  }

        return (inputs, outputs)

    def on_epoch_end(self):

        self.indexes = np.arange(self.len*self.batch_size)

        if self.shuffle == True:

            self.indexes = self.indexes.reshape(
                int(self.len), int(self.batch_size))
            np.random.shuffle(self.indexes)

            for i in range(self.len):
                np.random.shuffle(self.indexes[i])

            self.indexes = self.indexes.reshape(int(self.len*self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[int(index*self.batch_size):int((index+1)*self.batch_size)]


        batch_labels = [self.labels[int(k)] for k in indexes]
        batch_audios = [self.audios[int(k)] for k in indexes]

      

        return self.__data_generation(batch_labels, batch_audios)
