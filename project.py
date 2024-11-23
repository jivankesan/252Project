import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import os
from scipy.signal import resample

from scipy.signal import butter, sosfilt


def process_audio(filename):
    # Task 3.1: Read a sound file and find its sampling rate
    data, fs = sf.read("Audio/"+filename)
    print(f'Sampling rate: {fs} Hz')

    # Task 3.2: Check if the input sound is stereo or mono
    if data.ndim > 1:
        print(f'Input sound is stereo with shape {data.shape}')
        # Convert stereo to mono by averaging the two channels
        data_mono = np.mean(data, axis=1)
        print("Converted to mono")
    else:
        print('Input sound is mono')
        data_mono = data

    # Task 3.3: Play the sound
    print('Playing the original sound...')
    #sd.play(data_mono, fs)
    # sd.wait() 

    # Task 3.4: Write the sound to a new file
    output_folder = 'outputs'
    output_filename = os.path.join(output_folder, 'output_' + filename)

    # Ensure the 'outputs' folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Write the audio file
    sf.write(output_filename, data_mono, fs)
    print(f'Sound has been written to {output_filename}')

    # Task 3.5: Plot the sound waveform as a function of sample number
    plt.figure()
    plt.plot(data_mono)
    plt.title('Sound Waveform')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.show()

    # Task 3.6: Downsample the signal to 16 kHz if necessary
    target_fs = 16000
    if fs != target_fs:
        if fs < target_fs:
            print(f'Sampling rate is less than {target_fs} Hz. Please provide a higher sampling rate file.')
        else:
            print(f'Downsampling from {fs} Hz to {target_fs} Hz')
            resample_ratio = target_fs / fs
            number_of_samples = int(len(data_mono) * resample_ratio)
            data_mono = resample(data_mono, number_of_samples)
            fs = target_fs

    """
    # Task 3.7: Generate a 1 kHz cosine signal
    duration = len(data_mono) / fs
    t = np.linspace(0, duration, len(data_mono), endpoint=False)
    frequency = 1000  # 1 kHz
    cosine_signal = np.cos(2 * np.pi * frequency * t)

    # Play the generated cosine signal
    print('Playing the generated 1 kHz cosine signal...')
    sd.play(cosine_signal, fs)
    sd.wait()

    # Plot two cycles of the cosine waveform
    cycles = 2
    period = 1 / frequency
    samples_per_cycle = int(fs * period)
    total_samples = cycles * samples_per_cycle

    plt.figure()
    plt.plot(t[:total_samples], cosine_signal[:total_samples])
    plt.title('1 kHz Cosine Signal - First Two Cycles')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()
    """ 
    # Task 4: Create a bandpass filter bank
    num_channels = 8  # Define N channels as needed
    filters = create_bandpass_filters(fs, num_channels)

    # Task 5: Filter the sound with the filter bank
    filtered_signals = filter_with_bank(data_mono, filters)

    # Task 6: Plot the output signals of the lowest and highest frequency channels
    plot_filtered_signals(filtered_signals)

    # Task 7: Rectify the output signals of all bandpass filters
    rectified_signals = rectify_signals(filtered_signals)

    # Task 8: Detect envelopes using a lowpass filter with 400 Hz cutoff
    envelopes = detect_envelope(rectified_signals, fs)

    # Task 9: Plot the extracted envelope of the lowest and highest frequency channels
    plot_envelopes(envelopes)
    
    # Task 10: Combine the envelopes and play the new sound
    combined_envelope = np.sum(envelopes, axis=0)
    combined_envelope /= np.max(np.abs(combined_envelope))  # Normalize the combined envelope

    # Play the combined envelope as sound
    print('Playing the combined envelope sound...')
    sd.play(combined_envelope, fs)
    sd.wait()

    # Plot the combined envelope
    plt.figure()
    plt.plot(combined_envelope)
    plt.title('Combined Envelope')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.show()

# Task 3.8: Implement a function to filter the sound using a bandpass filter

def create_bandpass_filters(fs, num_channels=16, f_low=100, f_high=8000):
    # Divide the frequency range into `num_channels` equal-width bands
    freqs = np.linspace(f_low, f_high, num_channels + 1)
    filters = []
    for i in range(num_channels):
        # Design a bandpass filter for each channel
        low = freqs[i]
        high = freqs[i + 1]
        sos = butter(4, [low / (fs / 2), high / (fs / 2)], btype='bandpass', output='sos', fs=fs)
        filters.append(sos)
    return filters

def filter_with_bank(data, filters):
    """
    Apply a bank of bandpass filters to the input data.

    Args:
        data (numpy.ndarray): The input audio signal.
        filters (list): A list of second-order sections for each bandpass filter.

    Returns:
        list: A list of filtered signals, one for each bandpass filter.
    """
    filtered_signals = []
    for sos in filters:
        filtered_data = sosfilt(sos, data)
        filtered_signals.append(filtered_data)
    return filtered_signals
    
def plot_filtered_signals(filtered_signals, output_folder='outputs'):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(filtered_signals[0])
    plt.title('Lowest Frequency Channel')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(filtered_signals[-1])
    plt.title('Highest Frequency Channel')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    output_path = os.path.join(output_folder, 'filtered_signals.png')
    plt.savefig(output_path)
    print(f'Filtered signals plot saved to {output_path}')
    plt.close()
        
def rectify_signals(filtered_signals):
    rectified_signals = [np.abs(signal) for signal in filtered_signals]
    return rectified_signals

def lowpass_filter(data, fs, cutoff=400):
    sos = butter(4, cutoff / (fs / 2), btype='low', output='sos')
    return sosfilt(sos, data)

def detect_envelope(rectified_signals, fs):
    envelopes = [lowpass_filter(signal, fs) for signal in rectified_signals]
    return envelopes


def plot_envelopes(envelopes, output_folder='outputs'):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(envelopes[0])
    plt.title('Envelope of Lowest Frequency Channel')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(envelopes[-1])
    plt.title('Envelope of Highest Frequency Channel')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    output_path = os.path.join(output_folder, 'envelopes.png')
    plt.savefig(output_path)
    print(f'Envelopes plot saved to {output_path}')
    plt.close()
    
    
if __name__ == '__main__':
    file = input("What is the file name for the audio: ")
    process_audio(file)