import librosa  # for audio processing
import librosa.display
import pandas as pd
import os


class DataLoader:

    def __init__(self, train_audio_folder, train_script_file_path, sr=8000):
        self.train_audio_folder = train_audio_folder
        self.train_script_file_path = train_script_file_path
        self.sr = sr

    def extract_transcription_and_labels(self):

        file_path = self.train_script_file_path

        transcriptions = []
        with open(file_path) as f:
            line = f.readline()
            while (line):
                transcriptions.append(line)
                line = f.readline()

        label_trans_dict = dict()
        for trans in transcriptions:
            text = trans.replace("<s>", "").replace("</s>", "")
            text = text.replace("</s>", "")
            text = text.strip()

            label = text.split()[-1]

            try:
                label = label.replace("(", "")
                label = label.replace(")", "")
            except:
                pass

            translation = text.split()[:-1]
            translation = ' '.join(translation)

            label_trans_dict[label] = translation

        return label_trans_dict

    def extract_audio(self, max_lenght=10000, sr=8000):

        path = self.train_audio_folder

        wav_dict = dict()
        wav_paths = self.get_all_wav_paths(path)
        for path in wav_paths:
            if len(list(wav_dict.keys())) >= max_lenght:
                break
            wav, sample_rate = librosa.load(
                self.train_audio_folder+path, sr=self.sr)
            dur = float(len(wav)/sample_rate)
            channel = len(wav.shape)
            label = path.split(".")[0]
            wav_dict[label] = (wav, dur, channel, sample_rate)

        return wav_dict

    def get_all_wav_paths(self, folder_path):
        return os.listdir(folder_path)

    def create_meta_data(self, transcripton_obj, audo_obj):
        translations = []
        durations = []
        labels = []
        channels = []
        srs = []
        for k in audo_obj.keys():
            trans = transcripton_obj[k]
            label = k

            duration = audo_obj[k][1]
            channel = audo_obj[k][2]
            sr = audo_obj[k][3]

            translations.append(trans)
            durations.append(duration)
            labels.append(label)
            channels.append(channel)
            srs.append(sr)

            m_df = pd.DataFrame()
            m_df["translation"] = translations
            m_df["label"] = labels
            m_df["channel"] = channels
            m_df["sample_rate"] = srs
            m_df["duration"] = durations

        return m_df
