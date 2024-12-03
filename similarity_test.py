import os
import glob
import soundfile as sf
from pystoi import stoi
import librosa
import numpy as np

# Paths to the folders
comparisons_folder = './comparisons'
outputs_folder = './outputs'
output_file = 'stoi_metrics.txt'

# Get list of files in both folders
comparison_files = glob.glob(os.path.join(comparisons_folder, '*.wav'))
output_files = glob.glob(os.path.join(outputs_folder, '*.wav'))

# Create a dictionary to store the comparison files with their prefixes
comparison_dict = {}
for file in comparison_files:
    prefix = os.path.basename(file).split('_')[0]
    comparison_dict[prefix] = file

# Open the output file to write the STOI metrics
with open(output_file, 'w') as f:
    for file in output_files:
        prefix = os.path.basename(file).split('_')[0]
        if prefix in comparison_dict:
            # Read the audio files
            ref_signal, ref_rate = sf.read(comparison_dict[prefix])
            deg_signal, deg_rate = sf.read(file)
            
            # Convert to mono if necessary
            if len(ref_signal.shape) > 1:
                ref_signal = np.mean(ref_signal, axis=1)
            if len(deg_signal.shape) > 1:
                deg_signal = np.mean(deg_signal, axis=1)
            
            # Ensure the sample rates are the same
            if ref_rate != deg_rate:
                print(f"Sample rates do not match for {prefix}: {ref_rate} vs {deg_rate}, Downsampling...")
                # Downsample to 16kHz if necessary
                target_rate = 16000
                if ref_rate != target_rate:
                    ref_signal = librosa.resample(ref_signal, orig_sr=ref_rate, target_sr=target_rate)
                    ref_rate = target_rate
                if deg_rate != target_rate:
                    deg_signal = librosa.resample(deg_signal, orig_sr=deg_rate, target_sr=target_rate)
                    deg_rate = target_rate

            # Ensure the signals have the same length
            min_length = min(len(ref_signal), len(deg_signal))
            ref_signal = ref_signal[:min_length]
            deg_signal = deg_signal[:min_length]

            # Calculate the STOI metric
            try:
                stoi_metric = stoi(ref_signal, deg_signal, ref_rate, extended=False)
                # Write the result to the file
                f.write(f"{prefix}: {stoi_metric}\n")
            except Exception as e:
                print(f"Error calculating STOI for {prefix}: {e}")