import numpy as np
import librosa  # for audio processing
import librosa.display


class Data_Augmentation:

    def __init__(self):
        pass

    def add_noise(self, audio):
        data = audio
        noise_factor = np.random.uniform(0.01, 0.04)
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        augmented_data = augmented_data.astype(type(data[0]))
        return augmented_data

    def change_pitch(self, audio, sampling_rate):
        data = audio
        pitch_factor = np.random.uniform(-2, 2)

        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    def change_speed(self, audio):

        data = audio
        speed_factor = np.random.uniform(0.2, 2)
        return librosa.effects.time_stretch(data, speed_factor)

    def shift_signal(self, data, sampling_rate, shift_sec, shift_direction):

        shift = int(sampling_rate * shift_sec)
        if shift_direction == 'right':
            shift = -shift
        else:
            shift = shift

        augmented_data = np.roll(data, shift)

        return augmented_data
