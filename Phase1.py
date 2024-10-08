import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample

def process_audio(filename):
    # Task 3.1: Read a sound file and find its sampling rate
    data, fs = sf.read(filename)
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
    sd.play(data_mono, fs)
    sd.wait() 

    # Task 3.4: Write the sound to a new file
    output_filename = 'output_' + filename
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
    
    
if __name__ == '__main__':
    file = input("What is the file name for the audio: ")
    process_audio(file)