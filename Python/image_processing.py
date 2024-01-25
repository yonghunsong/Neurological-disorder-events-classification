import os
os.environ['LIBROSA_NO_CYTHON'] = '1'
import pywt
import librosa
import librosa.display
import numpy as np
from sklearn.preprocessing import scale
from PIL import Image


def Mel_Spectrogram(wav_file):

    y = wav_file
    sr = 6400

    input_nfft = 512
    input_stride = 64

    Mel_spectrogram = librosa.feature.melspectrogram(y=y, n_mels=128, n_fft=input_nfft, hop_length=input_stride)
    Mel_log_spectrogram = librosa.power_to_db(Mel_spectrogram, ref=np.max)

    return Mel_log_spectrogram



def Mel_Spectrogram_3CH(wav_file):

    y = wav_file
    sr = 6400

    input_nfft = 512
    input_stride = 64

    Mel_spectrogram = librosa.feature.melspectrogram(y=y, n_mels=128, n_fft=input_nfft, hop_length=input_stride)
    Mel_log_spectrogram = librosa.power_to_db(Mel_spectrogram, ref=np.max)

    image = mel_spectrogram_to_image(Mel_log_spectrogram)
    image = image.convert('RGB')
    Mel_log_spectrogram_resize = image.resize((224, 224), Image.ANTIALIAS)
    Mel_log_spectrogram_resize = change_image_dimensions(Mel_log_spectrogram_resize)

    return Mel_log_spectrogram_resize

def mel_spectrogram_to_image(mel_spectrogram):

    mel_spectrogram = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))
    mel_image = Image.fromarray((mel_spectrogram * 255).astype(np.uint8))

    return mel_image


def change_image_dimensions(image):

    changed_image = np.transpose(image, (2, 0, 1))
    return changed_image


def Spectrogram_3CH(wav_file):

    y = wav_file
    sr = 6400

    window_length = 128
    hop_length = 64
    window = 'hamm'
    n_fft = 512

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=window_length, window=window)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)

    image = mel_spectrogram_to_image(log_spectrogram)
    image = image.convert('RGB')
    log_spectrogram_resize = image.resize((224, 224), Image.ANTIALIAS)
    log_spectrogram_resize = change_image_dimensions(log_spectrogram_resize)
    
    return log_spectrogram_resize


def Spectrogram(wav_file):

    y = wav_file
    sr = 6400

    window_length = 512
    hop_length = 64
    window = 'hamm'
    n_fft = 512

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=window_length, window=window)
    magnitude = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    return log_spectrogram






