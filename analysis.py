import os
import json
import librosa
import numpy as np
from scipy.stats import linregress
import google.genai as genai
import config

def analyze_and_generate_data(file_path):
    """
    Analyzes an audio file and generates AI-based commentary.
    
    Args:
        file_path (str): The path to the audio file.
        
    Returns:
        tuple: A tuple containing a list of commentary data and the report narrative,
               or (None, None) if an error occurs.
    """
    client = genai.Client(api_key=config.API_KEY)

    y, sr = librosa.load(file_path, mono=True)
    frame_length = 2048
    hop_length = 512
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    
    mean_chroma_std = np.std(np.mean(chroma, axis=1))
    mean_zcr = np.mean(zcr)
    tonality = "tonal" if mean_zcr < 0.15 and mean_chroma_std > 0.2 else "atonal or non-traditional"

    frame_duration = librosa.get_duration(y=y, sr=sr) / chroma.shape[1]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    commentary_interval_seconds = 10
    step = max(1, int(commentary_interval_seconds / frame_duration))
    
    downsampled_times = times[::step]
    
    analysis_points = []
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i in range(len(downsampled_times)):
        current_time = downsampled_times[i]
        time_segment = times[i*step:(i+1)*step]
        rms_segment = rms[i*step:(i+1)*step]
        cent_segment = cent[i*step:(i+1)*step]
        
        rms_slope = 0
        cent_slope = 0
        if len(time_segment) > 1:
            rms_slope, _, _, _, _ = linregress(time_segment, rms_segment)
            cent_slope, _, _, _, _ = linregress(time_segment, cent_segment)

        strongest_pitch_index = np.argmax(chroma[:, i*step])
        strongest_pitch = pitch_classes[strongest_pitch_index]
        
        mfcc_mean_1_3 = np.mean(mfccs[:3, i*step])

        analysis_points.append(
            f"Time: {current_time:.2f}s, "
            f"RMS (Loudness): {rms[i*step]:.4f} (Trend: {rms_slope:.4f}), "
            f"Spectral Centroid (Brightness): {cent[i*step]:.2f} (Trend: {cent_slope:.2f}), "
            f"ZCR (Noisiness): {zcr[i*step]:.4f}, "
            f"Key: {strongest_pitch}, "
            f"MFCCs (Timbre): {mfcc_mean_1_3:.2f}"
        )
    
    audio_file_name = os.path.basename(file_path)
    
    prompt = f"""
    You are an expert audio commentator and music analyst. Your task is to provide two types of analysis for a music track named '{audio_file_name}'.
    The music has been identified as {tonality}.
    
    Part 1: Commentary for Visualization
    Provide time-stamped commentary in a JSON array. Each entry should have a "time" (in seconds) and a "commentary" string. The commentary should be educational and describe musical events based on the provided data.
    
    Part 2: High-Level Narrative for Textual Report
    Provide a high-level, narrative analysis in a paragraph format. This analysis should describe the overall dynamics (RMS), timbre (Spectral Centroid, ZCR, MFCCs), and rhythmic structure.
    
    The final output must be a single JSON object with two keys: "commentary_data" (containing the JSON array from Part 1) and "report_narrative" (containing the string from Part 2).
    
    Musicological Definitions:
    - RMS (Loudness): Overall intensity. A rising RMS indicates a crescendo.
    - Spectral Centroid (Brightness): Perceived brightness. A rising value means the sound is getting brighter.
    - ZCR (Noisiness): The degree of noisiness or percussiveness. A higher value suggests a noisier texture.
    - Key: The strongest detected pitch class. Changes may indicate harmonic shifts.
    - MFCCs (Timbre): Mel-Frequency Cepstral Coefficients. They represent the tonal color or texture of the sound. Changes often indicate new instruments or vocal events.
    
    Here is the audio analysis data:
    {analysis_points}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        if not response.text:
            print("Error: Gemini API response text is empty or None.")
            return None, None
        
        clean_json_str = response.text.strip().replace("```json", "").replace("```", "")
        response_data = json.loads(clean_json_str)
        
        commentary_data = response_data.get("commentary_data", [])
        report_narrative = response_data.get("report_narrative", "AI narrative could not be generated.")

        print("Data generated successfully!")
        return commentary_data, report_narrative
    except Exception as e:
        print(f"An error occurred during API call or JSON parsing: {e}")
        return None, None