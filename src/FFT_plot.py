import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series(data):
    plt.plot(data)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data')
    plt.show()


def plot_frequency_spectrum(data, sampling_rate):
    spectrum = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=1 / sampling_rate)
    plt.plot(frequencies, np.abs(spectrum))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.show()

def plot_summation_sinewave(data):
    wave_sum = [0]*200
    for i in range(len(data)):
        frequency_1 = data[i]
        t = np.linspace(0, 5, 200)
        phase_1 = 0
        amplitude_1 = 100
        wave_i = amplitude_1 * np.sin(2 * np.pi * frequency_1 * t + phase_1)
        wave_sum = wave_sum + wave_i
    t = np.linspace(0, 5, 200)
    plt.plot(t, wave_sum)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Summation of Sine Waves')
    plt.show()

# Example data
# data = [101 ,3628 ,1999 ,2392 ,1997 ,1996 ,3137 ,102 ,0 ,0  ,0  ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]

raw_data = pd.read_excel("/opt/project/dataset/result_predictions_token_flooding.xlsx")
for i in range(10):
    sep_data = raw_data["caption"][i].split()
    cleaned_data = [data for data in sep_data if data not in ["[", "]", "\n", " ", "0]"]]
    final_data = [int(x) for x in cleaned_data]
    # plot_frequency_spectrum(final_data, sampling_rate=1)  # Assuming a sampling rate of 1 (1 unit per time step
    plot_summation_sinewave(final_data)

plt.savefig('/opt/project/tmp/SumWave_plotFL.jpg')
