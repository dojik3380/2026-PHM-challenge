import matplotlib.pyplot as plt
from nptdms import TdmsFile

file_path = r"C:\Users\kdj33\OneDrive\바탕 화면\PHM\data\Train\Train4_Vibration\Train4_Vibration\000001.tdms"

tdms = TdmsFile.read(file_path)

# CH1 선택
for group in tdms.groups():
    for channel in group.channels():
        if channel.name == "CH1":
            signal = channel[:]

plt.figure(figsize=(10,4))
plt.plot(signal[:5000])  # 일부만 보기
plt.title("Raw Vibration (CH1)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()

import numpy as np
from scipy.signal import stft

fs = 25600  # 샘플링레이트

f, t, Zxx = stft(signal, fs=fs, nperseg=256, noverlap=128)

plt.figure(figsize=(10,5))
plt.pcolormesh(t, f, np.log1p(np.abs(Zxx)), shading='gouraud')
plt.title("STFT Spectrogram")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.colorbar()
plt.show()

fft = np.fft.fft(signal)
freq = np.fft.fftfreq(len(signal), d=1/fs)

plt.plot(freq[:len(freq)//2], np.abs(fft[:len(fft)//2]))
plt.title("FFT Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()