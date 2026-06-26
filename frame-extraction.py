import cv2
import os

# Configuration - Update these paths
VIDEO_PATH = r"C:\Users\anguy\OneDrive - Kennesaw State University\Documents\AI_Training\KSU-Edge-AI-Navigation\test\videos\ksu_innovation_indoor.mp4"
OUTPUT_FOLDER = r"C:\Users\anguy\OneDrive - Kennesaw State University\Documents\AI_Training\KSU-Edge-AI-Navigation\test\images"
TARGET_IMAGE_COUNT = 120

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the video
video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    print(f"Error: Could not open video at {VIDEO_PATH}")
    exit()

# Get metadata
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = max(1, total_frames // TARGET_IMAGE_COUNT)

print(f"Processing {total_frames} frames. Extracting every {frame_interval} frames.")

extracted_count = 0
for i in range(total_frames):
    ret, frame = video.read()
    
    if not ret:
        break

    # Extract exactly 100 frames with high quality
    if i % frame_interval == 0 and extracted_count < TARGET_IMAGE_COUNT:
        filename = os.path.join(OUTPUT_FOLDER, f"scene_{extracted_count:03d}.jpg")
        
        # Save with 95% quality for research clarity
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        extracted_count += 1
        
        # Log progress every 10 images
        if extracted_count % 10 == 0:
            print(f"Progress: {extracted_count}/{TARGET_IMAGE_COUNT} images saved.")

# Release resources
video.release()
print(f"Extraction complete. {extracted_count} images saved to {OUTPUT_FOLDER}.")