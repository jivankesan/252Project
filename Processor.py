import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import os
from scipy.signal import butter, sosfilt, resample


def process_audio(filename, audio_folder='Audio', output_folder='outputs'):
    file_path = os.path.join(audio_folder, filename)
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return

    # Read audio file
    data, fs = sf.read(file_path)
    print(f'Processing file: {filename}')
    print(f'Sampling rate: {fs} Hz')
    
    # Check if stereo and convert to mono
    if data.ndim > 1:
        print(f'Input sound is stereo with shape {data.shape}')
        data_mono = np.mean(data, axis=1)
        print("Converted to mono")
    else:
        print('Input sound is mono')
        data_mono = data

    # Downsample to 16 kHz if necessary
    target_fs = 16000
    if fs != target_fs:
        if fs < target_fs:
            print(f'Sampling rate is less than {target_fs} Hz. Please provide a higher sampling rate file.')
        else:
            print(f'Downsampling from {fs} Hz to {target_fs} Hz')
            resample_ratio = target_fs / fs
            data_mono = resample(data_mono, int(len(data_mono) * resample_ratio))
            fs = target_fs

      
    # Create output folder for plots
    plot_folder = os.path.join(output_folder, f'plots_{os.path.splitext(filename)[0]}')
    os.makedirs(plot_folder, exist_ok=True)  

    # Plot original signal
    plt.figure()
    plt.plot(data_mono)
    plt.title('Original Signal')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    original_plot_path = os.path.join(plot_folder, 'original_signal.png')
    plt.savefig(original_plot_path)
    plt.close()
    print(f'Original signal plot saved to {original_plot_path}')



    # Create bandpass filter bank
    num_channels = 16
    freqs = np.linspace(100, 8000, num_channels + 1)  
    filters = create_bandpass_filters(fs, num_channels)

    # Filter the audio and process
    filtered_signals = filter_with_bank(data_mono, filters)
    rectified_signals = rectify_signals(filtered_signals)
    envelopes = detect_envelope(rectified_signals, fs)

    # Plot filtered signals
    plot_filtered_signals(filtered_signals, plot_folder)

    # Plot envelopes
    plot_envelopes(envelopes, plot_folder)

    # Generate cosine signals
    cosine_signals = generate_cosine_signals(envelopes, fs, freqs)

    # Perform amplitude modulation
    modulated_signals = amplitude_modulation(cosine_signals, envelopes)

    # Combine signals
    combined_signal = combine_signals(modulated_signals)

    # Save and plot final output
    play_and_save_audio(combined_signal, fs, output_folder, filename, plot_folder)


def iterate_audio_folder(audio_folder='Audio', output_folder='outputs'):
    if not os.path.exists(audio_folder):
        print(f"Audio folder '{audio_folder}' does not exist.")
        return

    # Create outputs folder
    os.makedirs(output_folder, exist_ok=True)

    # Process each file in the Audio directory
    for filename in os.listdir(audio_folder):
        if filename.lower().endswith(('.wav', '.flac')):
            process_audio(filename, audio_folder, output_folder)
        else:
            print(f"Skipping non-audio file: {filename}")



def create_log_bandpass_filters(fs, num_channels=16, f_low=100, f_high=8000, order=8, overlap_fraction=0.01):
    """
    Create a bank of logarithmically spaced bandpass filters with specified overlap,
    modeling the spacing used in cochlear implants.

    Args:
        fs (int): Sampling frequency in Hz.
        num_channels (int): Number of bandpass filters to create.
        f_low (float): Lowest cutoff frequency in Hz.
        f_high (float): Highest cutoff frequency in Hz.
        order (int): Order of the Butterworth filter.
        overlap_fraction (float): Fractional overlap between adjacent bands (0 to 1).

    Returns:
        list: A list of bandpass filters in second-order section (SOS) format.
    """
    # Calculate logarithmically spaced edge frequencies for the filters
    frequencies = np.logspace(np.log10(f_low), np.log10(f_high), num_channels + 1)
    
    filters = []
    for i in range(num_channels):
        # Original band edges
        low = frequencies[i]
        high = frequencies[i + 1]
        
        # Calculate bandwidth and overlap
        bandwidth = high - low
        overlap = bandwidth * overlap_fraction
        
        # Adjust the low and high frequencies to include overlap
        if i == 0:
            low_adj = low
        else:
            low_adj = low - overlap / 2

        if i == num_channels - 1:
            high_adj = high
        else:
            high_adj = high + overlap / 2

        # Ensure adjusted frequencies stay within the overall frequency range
        low_adj = max(f_low, low_adj)
        high_adj = min(f_high, high_adj)

        # Normalize frequencies to the Nyquist frequency
        low_norm = low_adj / (fs / 2)
        high_norm = high_adj / (fs / 2)

        # Check if the normalized frequencies are valid
        if not (0 < low_norm < high_norm < 1):
            print(f"Skipping invalid filter {i+1}: low_norm={low_norm}, high_norm={high_norm}")
            high_norm = min(high_norm, 1 - 1e-6)

        # Design the bandpass filter
        try:
            sos = butter(order, [low_norm, high_norm], btype='bandpass', output='sos')
            filters.append(sos)
            print(f"Created filter {i+1}: {low_adj:.2f} Hz to {high_adj:.2f} Hz")
        except ValueError as e:
            print(f"Error creating filter {i+1}: {e}")
            continue

    return filters


def create_bandpass_filters(fs, num_channels=16, f_low=100, f_high=8000, order=8, max_overlap=200):
    """
    Create a bank of bandpass filters with 16 subdivisions and capped overlap.

    Args:
        fs (int): Sampling frequency in Hz.
        num_channels (int): Number of bandpass filters to create.
        f_low (float): Lowest cutoff frequency in Hz.
        f_high (float): Highest cutoff frequency in Hz.
        order (int): Order of the Butterworth filter.
        max_overlap (float): Maximum overlap in Hz between adjacent filters.

    Returns:
        list: A list of bandpass filters in second-order section (SOS) format.
    """
    # Calculate base frequency range for each filter
    band_width = (f_high - f_low) / num_channels
    filters = []
    
    for i in range(num_channels):
        # Calculate the low and high cutoff for each filter
        low = f_low + i * band_width
        high = low + band_width

        # Cap overlap at max_overlap
        if i > 0:
            low -= min(max_overlap, band_width / 2)
        if i < num_channels - 1:
            high += min(max_overlap, band_width / 2)

        # Ensure bounds stay within f_low and f_high
        low = max(f_low, low)
        high = min(f_high, high)

        # Normalize frequencies to the Nyquist frequency
        low_norm = low / (fs / 2)
        high_norm = high / (fs / 2)

        # Skip invalid filters
        if not (0 < low_norm < high_norm < 1):
            high_norm = min(high_norm, 1 - 1e-6)

        # Design bandpass filter
        sos = butter(order, [low_norm, high_norm], btype='bandpass', output='sos')
        filters.append(sos)
        print(f"Created filter {i+1}: {low:.2f} Hz to {high:.2f} Hz")

    return filters

def filter_with_bank(data, filters):
    return [sosfilt(sos, data) for sos in filters]

def rectify_signals(filtered_signals):
    return [np.abs(signal) for signal in filtered_signals]

def lowpass_filter(data, fs, cutoff=400):
    sos = butter(8, cutoff / (fs / 2), btype='low', output='sos')
    return sosfilt(sos, data)

def detect_envelope(rectified_signals, fs):
    envelopes = [lowpass_filter(signal, fs) for signal in rectified_signals]
    return envelopes

def plot_filtered_signals(filtered_signals, plot_folder):
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
    output_path = os.path.join(plot_folder, 'filtered_signals.png')
    plt.savefig(output_path)
    plt.close()
    print(f'Filtered signals plot saved to {output_path}')

def plot_envelopes(envelopes, plot_folder):
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
    output_path = os.path.join(plot_folder, 'envelopes.png')
    plt.savefig(output_path)
    plt.close()
    print(f'Envelopes plot saved to {output_path}')

def generate_cosine_signals(envelopes, fs, freqs):
    cosine_signals = []
    for i, envelope in enumerate(envelopes):
        f_central = (freqs[i] + freqs[i + 1]) / 2
        print(f'Channel {i+1}: Central frequency = {f_central:.2f} Hz')
        t = np.linspace(0, len(envelope) / fs, len(envelope), endpoint=False)
        cosine_signal = np.cos(2 * np.pi * f_central * t)
        cosine_signals.append(cosine_signal)

    return cosine_signals

def amplitude_modulation(cosine_signals, envelopes):
    return [cosine * envelope for cosine, envelope in zip(cosine_signals, envelopes)]

def combine_signals(am_signals):
    combined_signal = np.sum(am_signals, axis=0)
    return combined_signal / np.max(np.abs(combined_signal))

def play_and_save_audio(signal, fs, output_folder, filename, plot_folder):
    base_filename = os.path.splitext(filename)[0]
    output_audio_path = os.path.join(output_folder, f'{base_filename}_output.wav')
    sf.write(output_audio_path, signal, fs)
    print(f'Combined audio signal saved to {output_audio_path}')
        
    # Plot the final output signal
    plt.figure()
    plt.plot(signal)
    plt.title('Combined Output Signal')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    output_plot_path = os.path.join(plot_folder, f'{base_filename}_output_plot.png')
    plt.savefig(output_plot_path)
    plt.close()
    print(f'Combined output signal plot saved to {output_plot_path}')



if __name__ == '__main__':
    iterate_audio_folder()