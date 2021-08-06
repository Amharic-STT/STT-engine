import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

class AudioUtil():
  
  # Load an audio file. Return the signal as a tensor and the sample rate
  
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

  def pad_trunc(aud,max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      print(sig.shape)
      return (sig, sr)  

  def resize_all(self, df, sm=10000):
    sig_list = []
    sr_list = []
    for i in range(len(df)):
      audio_path = df['key'][i]
      aud = AudioUtil.open(audio_path)
      sig, sr = AudioUtil.pad_trunc(aud,sm)
      sig_list.append(sig)
      sr_list.append(sr)
    return sig_list, sr_list

  def play_audio(self, samples, sample_rate):
      return ipd.Audio(samples, rate=sample_rate)    




