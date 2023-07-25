import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Sine wave parameters
sampling_rate = 1000  # Sampling rate in Hz
frequency = 100  # Frequency of the sine wave in Hz
amplitude = 1  # Amplitude of the sine wave

# Generate the sine wave signal
t = np.arange(0, 1, 1/sampling_rate)
sine_wave1 = amplitude * np.sin(2 * np.pi * 202 * t)
sine_wave2 = amplitude * np.sin(2 * np.pi * frequency * t)
sine_wave3 = amplitude * np.sin(2 * np.pi * 3000 * t)
sine_wave4 = amplitude * np.sin(2 * np.pi * 201 * t)
sine_wave5 = amplitude * np.sin(2 * np.pi * 8088 * t)
sine_wave6 = amplitude * np.sin(2 * np.pi * 455 * t)
sine_wave7 = amplitude * np.sin(2 * np.pi * 1988 * t)
sine_wave = sine_wave1 + sine_wave2 + sine_wave3 + sine_wave4 + sine_wave5 + sine_wave6 + sine_wave7
# Design the low-pass filter
cutoff_freq = 150  # Cutoff frequency of the low-pass filter in Hz
order = 4  # Filter order
b, a = signal.butter(order, cutoff_freq/(sampling_rate/2), btype='low')

# Apply the filter to the sine wave
filtered_signal = signal.lfilter(b, a, sine_wave)

# Plot the original sine wave and the filtered signal
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t, sine_wave)
plt.title('Original Sine Wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal)
plt.title('Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig('/opt/project/tmp/Test555.jpg')
plt.show()
