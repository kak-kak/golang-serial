import numpy as np
import cv2
import json
from datetime import datetime

def generate_wave_data(num_sequences=1, num_frames=100, size=(400, 400), line_thickness=1, wave_start_frame=50, noise_level=0.1, wave_intensity=1.0):
    sequences = []
    
    for seq in range(num_sequences):
        frames = []
        
        origin_x = np.random.randint(0, size[0])
        origin_y = np.random.randint(0, size[1])
        
        # Create a random background for this sequence
        background = np.zeros(size, dtype=np.float32)
        for _ in range(5):  # Number of rectangles
            top_left = (np.random.randint(0, size[0]//2), np.random.randint(0, size[1]//2))
            bottom_right = (np.random.randint(size[0]//2, size[0]), np.random.randint(size[1]//2, size[1]))
            intensity = np.random.uniform(0.3, 0.7)  # Intensity of the background rectangle
            cv2.rectangle(background, top_left, bottom_right, intensity, thickness=-1)
        
        y, x = np.ogrid[:size[0], :size[1]]
        distance = np.sqrt((x - origin_x) ** 2 + (y - origin_y) ** 2)

        for frame in range(num_frames):
            frame_data = background.copy()
            
            if frame >= wave_start_frame:
                radius = (frame - wave_start_frame) * 2
                
                mask = np.abs(distance - radius) < line_thickness
                frame_data[mask] = wave_intensity
                
                noise_mask = (np.random.rand(*size) < 0.05) & mask
                frame_data[noise_mask] = np.random.rand(np.sum(noise_mask)) * wave_intensity

            frame_data += noise_level * np.random.rand(*size)
            flicker_noise = (np.random.rand(*size) < noise_level) * np.random.rand(*size)
            frame_data += flicker_noise
            frame_data = np.clip(frame_data, 0, 1)
            
            frames.append(frame_data)
        
        sequences.append(frames)
    
    return sequences

def save_sequences_as_videos(sequences, size=(400, 400), params=None):
    json_filename = 'wave_sequence_metadata.json'
    
    # Load existing metadata if it exists
    try:
        with open(json_filename, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = []

    for i, sequence in enumerate(sequences):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        avi_filename = f'wave_sequence_{timestamp}_{i}.avi'
        
        out = cv2.VideoWriter(avi_filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (size[1], size[0]), False)
        for frame in sequence:
            frame_normalized = (frame * 255).astype(np.uint8)
            out.write(frame_normalized)
        out.release()
        
        # Append new metadata
        sequence_metadata = {
            "filename": avi_filename,
            "timestamp": timestamp,
            "parameters": params
        }
        metadata.append(sequence_metadata)
    
    # Save updated metadata
    with open(json_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

# Parameters for generating wave data
params = {
    "num_sequences": 10,
    "num_frames": 100,
    "size": (400, 400),
    "line_thickness": 1,
    "wave_start_frame": 50,
    "noise_level": 0.2,
    "wave_intensity": 0.3
}

# Generate the data
wave_data = generate_wave_data(**params)

# Save the data as video files and update metadata
save_sequences_as_videos(wave_data, size=params["size"], params=params)

print("Wave data generation and saving complete.")
