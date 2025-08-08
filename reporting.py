import os
import datetime
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def generate_textual_report(file_path, report_narrative, base_filename):
    """
    Generates a human-readable textual report from audio analysis data.
    
    Args:
        file_path (str): Path to the audio file.
        report_narrative (str): The AI-generated high-level narrative.
        base_filename (str): The base name for the output file.
    """
    y, sr = librosa.load(file_path, mono=True)
    frame_length = 2048
    hop_length = 512
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    onset_sf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
    
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    data = {
        'Time_Seconds': times,
        'RMS_Energy': rms,
        'Spectral_Centroid': cent,
        'ZCR': zcr,
        'Novelty_Curve': onset_sf,
    }
    for i in range(mfccs.shape[0]):
        data[f'MFCC_{i+1}'] = mfccs[i]

    df = pd.DataFrame(data)
    
    global_stats = {
        'RMS_Energy': {'mean': df['RMS_Energy'].mean(), 'std': df['RMS_Energy'].std()},
        'Spectral_Centroid': {'mean': df['Spectral_Centroid'].mean(), 'std': df['Spectral_Centroid'].std()},
        'ZCR': {'mean': df['ZCR'].mean(), 'std': df['ZCR'].std()},
        'Novelty_Curve': {'mean': df['Novelty_Curve'].mean(), 'std': df['Novelty_Curve'].std()}
    }

    report_output_file = f'{base_filename}_Analysis_Report.txt'
    
    with open(report_output_file, 'w', encoding='utf-8') as f:
        f.write(f"Quantitative Musicological Analysis Report\n")
        f.write(f"Piece: {os.path.basename(file_path)}\n")
        f.write(f"Duration: {df['Time_Seconds'].max():.2f} seconds\n")
        f.write(f"Date of Analysis: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("--- Global Feature Analysis (AI Interpreted) ---\n")
        f.write(f"This report is based on an analysis of audio features including RMS (loudness), Spectral Centroid (brightness), "
                f"ZCR (noisiness), and MFCCs (timbral qualities).\n\n")

        f.write(report_narrative)
        f.write("\n\n")

        f.write("--- Global Statistics ---\n")
        for feature, stats in global_stats.items():
            f.write(f"  {feature}:\n")
            f.write(f"    Mean: {stats['mean']:.4f}\n")
            f.write(f"    Standard Deviation: {stats['std']:.4f}\n")
        f.write("\n")

        f.write("--- Indications of Formal Boundaries ---\n")
        novelty_peaks_indices, _ = find_peaks(df['Novelty_Curve'],
                                              height=np.mean(df['Novelty_Curve']) + np.std(df['Novelty_Curve']) * 1.0,
                                              prominence=0.1)
        novelty_peak_times = df['Time_Seconds'].iloc[novelty_peaks_indices]
        f.write("### Potential Major Onsets/Changes (Novelty Curve Peaks)\n")
        if not novelty_peak_times.empty:
            f.write("  Moments of significant spectral change or 'newness' are indicated at:\n")
            for t in novelty_peak_times:
                f.write(f"  - {t:.2f} seconds\n")
        else:
            f.write("  No significant peaks found above the current threshold, indicating a continuously evolving or highly homogeneous texture.\n")
    
    print(f"\nAnalysis report saved to {report_output_file}")

def generate_files(file_path, base_filename):
    """
    Generates a CSV file and feature plots from audio analysis data.
    
    Args:
        file_path (str): Path to the audio file.
        base_filename (str): The base name for the output files.
    """
    y, sr = librosa.load(file_path, mono=True)
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    onset_sf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    data = {
        'Time_Seconds': times,
        'RMS_Energy': rms,
        'Spectral_Centroid': cent,
        'ZCR': zcr,
        'Novelty_Curve': onset_sf,
    }
    for i in range(mfccs.shape[0]):
        data[f'MFCC_{i+1}'] = mfccs[i]

    df = pd.DataFrame(data)

    csv_output_file = f'{base_filename}_Feature_Data.csv'
    df.to_csv(csv_output_file, index=False)
    print(f"Feature data saved to {csv_output_file}")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Quantitative Analysis of - {os.path.basename(file_path)}', fontsize=16)

    axes[0].plot(df['Time_Seconds'], df['RMS_Energy'], color='darkblue', alpha=0.8)
    axes[0].set_title('RMS Energy (Loudness Profile)', fontsize=12)
    axes[0].set_ylabel('RMS (Normalized)', fontsize=10)
    axes[0].grid(True)

    axes[1].plot(df['Time_Seconds'], df['Novelty_Curve'], color='darkgreen', alpha=0.8)
    axes[1].set_title('Novelty Curve (Onset Strength Function)', fontsize=12)
    axes[1].set_ylabel('Novelty Value', fontsize=10)
    axes[1].grid(True)

    axes[2].plot(df['Time_Seconds'], df['Spectral_Centroid'], color='darkred', alpha=0.8)
    axes[2].set_title('Spectral Centroid (Brightness)', fontsize=12)
    axes[2].set_ylabel('Frequency (Hz)', fontsize=10)
    axes[2].grid(True)

    axes[3].plot(df['Time_Seconds'], df['ZCR'], color='purple', alpha=0.8)
    axes[3].set_title('Zero-Crossing Rate (Noisiness/Percussiveness)', fontsize=12)
    axes[3].set_ylabel('ZCR', fontsize=10)
    axes[3].set_xlabel('Time (Seconds)', fontsize=12)
    axes[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_output_file = f'{base_filename}_Feature_Plots.png'
    plt.savefig(plot_output_file)
    print(f"Plots saved as {plot_output_file}")