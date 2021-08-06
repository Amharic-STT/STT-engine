from tokenizer import Tokenizer
import numpy as np


class DataGenerator:

    def __init__(self, translations, audios, batch_size):
        self.audios = audios
        self.translations = translations
        self.batch_size = batch_size

        self.tokenizer = Tokenizer(translations)
        self.int_to_char, self.char_to_int = self.tokenizer.build_dict()

        self.cur_index = 0

    def encode_text(self, translations):
        encoded_trans = []

        for t in translations:
            encoded = self.tokenizer.encode_text(t, self.char_to_int)
            encoded_trans.append(encoded)

        return encoded_trans

    def get_max_len(self, items):
        maximum = 0
        for i in items:
            if len(i) > maximum:
                maximum = len(i)

        return maximum

    def get_batch(self, index):

        batch_translations = self.translations[index: index + self.batch_size]
        batch_audios = self.audios[index: index + self.batch_size]

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
            label_length[ind, ] = len(trans)

            padded_audio = np.pad(
                audio, (0, maximum_audio_len - len(audio)), mode='constant', constant_values=0)
            padded_audios_np[ind, ] = padded_audio
            input_length[ind, ] = len(audio)

            ind += 1

        outputs = {'ctc': np.zeros([self.batch_size])}
        inputs = {'the_input': padded_audios_np,
                  'the_labels': encoded_trans_np,
                  'input_length': input_length,
                  'label_length': label_length
                  }

        return (inputs, outputs)

    def get_next_batch(self):

        batch_data = self.get_batch(self.cur_index)
        self.cur_index += self.batch_size

        return batch_data
