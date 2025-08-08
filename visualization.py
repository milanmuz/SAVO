import pygame
import librosa
import numpy as np
import sys
import imageio
import datetime
from utils import wrap_text

def run_visualization(audio_file_path, commentary_data_list, base_filename):
    """
    Runs the Pygame visualization and exports a video file.
    
    Args:
        audio_file_path (str): The path to the audio file.
        commentary_data_list (list): A list of dictionaries with commentary.
        base_filename (str): The base name for the output video file.
    """
    y, sr = librosa.load(audio_file_path, mono=True)
    audio_duration = librosa.get_duration(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).flatten()
    rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    
    chroma_duration_s = librosa.get_duration(y=y, sr=sr)
    chroma_frame_duration = chroma_duration_s / chroma.shape[1]
    
    commentary_data = {item['time']: item['commentary'] for item in commentary_data_list}
    commentary_times = sorted(commentary_data.keys())

    pygame.init()
    screen_width, screen_height = 1000, 700
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Triple-Layer Audio Visualization")
    clock = pygame.time.Clock()

    background_color = (0, 0, 0)
    label_color = (200, 200, 200)
    timeline_color = (150, 150, 150)
    
    spec_height = 175
    vu_meter_height = 175
    chroma_height = 175
    commentary_height = 150
    timeline_height = 25

    spec_surface = pygame.Surface((screen_width, spec_height))
    spec_surface.fill(background_color)
    vu_meter_surface = pygame.Surface((screen_width, vu_meter_height))
    vu_meter_surface.fill(background_color)
    chroma_surface = pygame.Surface((screen_width, chroma_height))
    chroma_surface.fill(background_color)
    commentary_surface = pygame.Surface((screen_width, commentary_height))
    commentary_surface.fill(background_color)
    timeline_surface = pygame.Surface((screen_width, timeline_height))
    timeline_surface.fill(background_color)

    num_spec = spec_db.shape[0]
    num_chroma = chroma.shape[0]

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    font = pygame.font.Font(None, 18)
    commentary_font = pygame.font.Font(None, 24)
    
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    
    video_writer = imageio.get_writer(f'{base_filename}_visualization.mp4', fps=int(1.0 / chroma_frame_duration))

    running = True
    current_commentary = ""
    last_commentary_time = -1
    last_seconds_mark = -1
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        current_time = pygame.mixer.music.get_pos() / 1000.0
        
        if not pygame.mixer.music.get_busy():
            running = False
            continue

        frame_index = int(np.floor(current_time / chroma_frame_duration))
        if frame_index >= len(rms_normalized):
            frame_index = len(rms_normalized) - 1
        
        if frame_index >= chroma.shape[1]:
            frame_index = chroma.shape[1] - 1
            
        current_chroma_data = chroma[:, frame_index]
        current_spec_data = spec_db[:, frame_index]
        current_rms_value = rms_normalized[frame_index]

        # Shift old pixels to the left and draw new ones on the right
        spec_surface.blit(spec_surface, (-1, 0))
        for i in range(num_spec):
            intensity = (current_spec_data[i] + 80) / 80
            gray_value = int(intensity * 255)
            color = (gray_value, gray_value, gray_value)
            pygame.draw.rect(spec_surface, color, (screen_width - 1, (num_spec - 1 - i) * (spec_height / num_spec), 1, spec_height / num_spec))

        vu_meter_surface.blit(vu_meter_surface, (-1, 0))
        vu_meter_surface.fill(background_color, (screen_width - 1, 0, 1, vu_meter_height))
        num_bars = 20
        lit_bars = int(current_rms_value * num_bars)
        bar_height = vu_meter_height / num_bars
        for i in range(num_bars):
            if i < lit_bars:
                if i < num_bars * 0.5:
                    color = (0, 255, 0)
                elif i < num_bars * 0.8:
                    color = (255, 255, 0)
                else:
                    color = (255, 0, 0)
                y_pos = vu_meter_height - (i + 1) * bar_height
                pygame.draw.rect(vu_meter_surface, color, (screen_width - 1, y_pos, 1, bar_height))
        
        chroma_surface.blit(chroma_surface, (-1, 0))
        for i in range(num_chroma):
            intensity = current_chroma_data[i]
            color_val = int(intensity * 255)
            bar_color = pygame.Color(color_val, 0, color_val)
            pygame.draw.rect(chroma_surface, bar_color, (screen_width - 1, i * (chroma_height / num_chroma), 1, chroma_height / num_chroma))

        # Handle commentary updates
        for time_stamp in commentary_times:
            if time_stamp > last_commentary_time and current_time >= time_stamp:
                current_commentary = commentary_data[time_stamp]
                last_commentary_time = time_stamp
                break
        
        commentary_surface.fill(background_color)
        
        wrapped_lines = wrap_text(current_commentary, commentary_font, screen_width - 20)
        
        line_spacing = commentary_font.get_height() + 5
        text_y_pos = (commentary_height - len(wrapped_lines) * line_spacing) // 2

        for line in wrapped_lines:
            commentary_text = commentary_font.render(line.strip(), True, label_color)
            text_rect = commentary_text.get_rect(center=(screen_width // 2, text_y_pos + commentary_font.get_height() // 2))
            commentary_surface.blit(commentary_text, text_rect)
            text_y_pos += line_spacing
        
        # Handle timeline updates
        timeline_surface.blit(timeline_surface, (-1, 0))
        timeline_surface.fill(background_color, (screen_width - 1, 0, 1, timeline_height))
        timeline_center_y = timeline_height // 2

        current_seconds = int(current_time)
        if current_seconds > last_seconds_mark:
            # Draw a line and number for every 5 seconds
            if current_seconds % 5 == 0:
                pygame.draw.line(timeline_surface, timeline_color, (screen_width - 1, timeline_center_y - 5), (screen_width - 1, timeline_center_y + 5), 1)
                
                minutes = int(current_seconds // 60)
                seconds = int(current_seconds % 60)
                time_label_text = f"{minutes:02d}:{seconds:02d}"

                time_label = font.render(time_label_text, True, label_color)
                # Adjust position to align with the vertical mark and be fully visible
                timeline_surface.blit(time_label, (screen_width - time_label.get_width() - 5, timeline_center_y - (time_label.get_height() // 2) + 1))
        
        last_seconds_mark = current_seconds

        screen.fill(background_color)
        
        # Blit surfaces in the new order and positions
        y_offset = 0
        screen.blit(spec_surface, (0, y_offset))
        y_offset += spec_height
        screen.blit(vu_meter_surface, (0, y_offset))
        y_offset += vu_meter_height
        screen.blit(chroma_surface, (0, y_offset))
        y_offset += chroma_height
        
        # Move timeline up to reduce space
        timeline_offset_y = 5 
        screen.blit(timeline_surface, (0, y_offset - timeline_offset_y))
        y_offset += timeline_height - timeline_offset_y
        
        # Adjusted commentary position
        commentary_offset_y = 5
        screen.blit(commentary_surface, (0, y_offset + commentary_offset_y))
        y_offset += commentary_height + commentary_offset_y
        
        # Draw separation lines
        y_line_1 = spec_height
        y_line_2 = spec_height + vu_meter_height
        y_line_3 = spec_height + vu_meter_height + chroma_height - timeline_offset_y
        
        pygame.draw.line(screen, label_color, (0, y_line_1), (screen_width, y_line_1), 3)
        pygame.draw.line(screen, label_color, (0, y_line_2), (screen_width, y_line_2), 3)
        pygame.draw.line(screen, label_color, (0, y_line_3), (screen_width, y_line_3), 3)

        # Draw labels
        freq_labels_text = ['125 Hz', '250 Hz', '500 Hz', '1k Hz', '2k Hz', '4k Hz', '8k Hz', '16k Hz']
        num_freq_labels = len(freq_labels_text)
        for i in range(num_freq_labels):
            y_pos = (spec_height / num_freq_labels) * (num_freq_labels - 1 - i)
            label = font.render(freq_labels_text[i], True, label_color)
            screen.blit(label, (5, y_pos - label.get_height() // 2))

        vu_labels = ['-6 dB', '-12 dB', '-24 dB']
        num_vu_labels = len(vu_labels)
        for i in range(num_vu_labels):
            y_pos = spec_height + vu_meter_height * (i + 1) / (num_vu_labels + 1)
            label = font.render(vu_labels[i], True, label_color)
            screen.blit(label, (5, y_pos - label.get_height() // 2))

        for i in range(num_chroma):
            label = font.render(pitch_classes[i], True, label_color)
            screen.blit(label, (5, spec_height + vu_meter_height + i * (chroma_height / num_chroma) + (chroma_height / num_chroma) // 2 - label.get_height() // 2))
        
        pygame.display.flip()

        frame = pygame.surfarray.array3d(screen)
        frame = np.rot90(frame, k=-1)
        frame = np.fliplr(frame)
        video_writer.append_data(frame)
        
        clock.tick(int(1.0 / chroma_frame_duration))

    video_writer.close()
    pygame.quit()
    sys.exit()