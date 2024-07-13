from pathlib import Path
import cv2
import os
from tqdm import tqdm
import re

# Function to extract the numerical part from the filename
def extract_number(filename):
    s = re.findall(r'\d+', filename)
    return int(s[0]) if s else -1

def create_video_from_images(image_dir, output_filename, fps=24):
    # Get all image file names
    image_files = [img for img in os.listdir(image_dir) if img.endswith(".png")]

    # Sort the files based on the numerical part extracted from each file name
    image_files = sorted(image_files, key=extract_number)
    
    if not image_files:
        print("No images found in the directory.")
        return
    
    # Determine the width and height from the first image
    sample_img = cv2.imread(os.path.join(image_dir, image_files[0]))
    height, width, _ = sample_img.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Change codec here
    out = cv2.VideoWriter(str(output_filename), fourcc, fps, (width, height))

    # Check if VideoWriter was successfully opened
    if not out.isOpened():
        print("Error: Could not open VideoWriter. Check codec and file path.")
        return
    
    for image_file in tqdm(image_files):
        img = cv2.imread(os.path.join(image_dir, image_file))
        out.write(img)  # Write frame to video
    
    # Release everything if job is finished
    out.release()
    print(f"Video written to {output_filename}")


if __name__ == "__main__":
    # Define the path to the directory containing the images
    FIGURES_DIR = Path("notebooks/animation_frames")
    OUTPUT_PATH = FIGURES_DIR / "horizontal_line.mp4"

    create_video_from_images(FIGURES_DIR, OUTPUT_PATH, fps=30)
