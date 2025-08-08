## SAVO - Sound Analysis Video Output

SAVO (Sound Analysis Video Output) is a powerful, Python-based tool that automates the process of analyzing an audio file and generating a comprehensive suite of outputs: a detailed textual report, static feature plots, a CSV of all feature data, and a dynamic video visualization.

The core of SAVO is its ability to combine traditional musicological analysis with AI-generated commentary. Using the `librosa` library, the program extracts key audio features such as RMS energy, spectral centroid, zero-crossing rate, and MFCCs. This data is then sent to the Gemini API, which acts as a music analyst to produce a high-level narrative and time-stamped commentary.

Example videos:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/lHAu7n2WzS8/0.jpg)](https://www.youtube.com/watch?v=lHAu7n2WzS8)


## Features

- Automated Audio Analysis: Extracts a wide range of audio features including RMS (loudness), Spectral Centroid (brightness), ZCR (noisiness), MFCCs (timbre), and a novelty curve (rhythmic changes).
- AI-Powered Commentary: Utilizes the Gemini API to interpret the audio analysis data and generate a time-stamped commentary and a high-level narrative report.
- Dynamic Video Visualization: Renders a multi-layered video using `pygame` that displays a real-time spectrogram, a VU meter, and a chroma visualization, all synchronized with the audio and the AI-generated commentary.
- Comprehensive Reporting: Produces three types of output files in addition to the video:
    - `_Analysis_Report.txt`: A human-readable report with global statistics, an AI-generated narrative, and a list of potential formal boundaries.
    - `_Feature_Data.csv`: A CSV file containing all the raw audio feature data for further analysis.
    - `_Feature_Plots.png`: A static image file with plots of the key audio features over time.
- Formatted Timeline: The video visualization includes a clean, synchronized timeline showing minutes and seconds in "MM:SS" format.

## Requirements

To run SAVO, you need Python installed on your system. You can install the required libraries using `pip`:

```bash
pip install pygame librosa numpy imageio pandas matplotlib google-generativeai

You will also need to set up a config.py file to hold your Google Gemini API key.
Example: config.py
API_KEY = "YOUR_GEMINI_API_KEY"

## Usage

To use SAVO, simply run the main.py script from your terminal and pass the path to your audio file as an argument. The program will handle all analysis and file generation automatically.

python main.py path/to/your/audio_file.wav

Upon completion, you will find the following files in the same directory:
audio_file_name_timestamp_visualization.mp4
audio_file_name_timestamp_Analysis_Report.txt
audio_file_name_timestamp_Feature_Data.csv
audio_file_name_timestamp_Feature_Plots.png

## Project Structure

main.py: The entry point for the entire program. It orchestrates the flow by calling the other modules.
analysis.py: Handles the audio feature extraction using librosa and communicates with the Gemini API to generate the commentary_data and report_narrative.
reporting.py: Generates the textual report (.txt), the feature data spreadsheet (.csv), and the plots (.png).
visualization.py: The core visualization module using pygame and imageio to create the video output with all the visual layers.
utils.py: Contains helper functions, such as wrap_text, used by other modules.
config.py: (Required) A file to store your Gemini API key.

