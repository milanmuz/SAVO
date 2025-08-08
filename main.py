import sys
import os
import datetime
from analysis import analyze_and_generate_data
from reporting import generate_textual_report, generate_files
from visualization import run_visualization

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_file_path = sys.argv[1]

    if not os.path.exists(audio_file_path):
        print(f"Error: The file '{audio_file_path}' was not found.")
        sys.exit(1)
        
    audio_file_name_base = os.path.splitext(os.path.basename(audio_file_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{audio_file_name_base}_{timestamp}"

    print("Step 1: Analyzing audio and generating commentary with AI...")
    commentary_data_list, report_narrative = analyze_and_generate_data(audio_file_path)
    
    if not commentary_data_list or not report_narrative:
        print("Could not generate all data. Exiting.")
        sys.exit(1)
    
    print("Step 2: Generating textual report and feature files...")
    generate_textual_report(audio_file_path, report_narrative, base_filename)
    generate_files(audio_file_path, base_filename)
    
    print("Step 3: Running audio visualization and exporting video...")
    run_visualization(audio_file_path, commentary_data_list, base_filename)
    
    print(f"\nProcess complete. Check the directory for the generated files:")
    print(f"- {base_filename}_Analysis_Report.txt")
    print(f"- {base_filename}_Feature_Data.csv")
    print(f"- {base_filename}_Feature_Plots.png")
    print(f"- {base_filename}_visualization.mp4")

if __name__ == "__main__":
    main()