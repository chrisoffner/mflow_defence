import cv2
from pdf2image import convert_from_path
from tqdm import tqdm
import numpy as np
from pathlib import Path
import re

# Set the directory containing the PDF files
pdf_directory = Path("figures/spiral_mflow_nrotations_1.2_successful_D")

# Set the output video file name
output_video = pdf_directory / "video4k.mp4"

# Function to extract number from filename
def extract_number(filename):
    match = re.search(r'\d+', filename.stem)
    return int(match.group()) if match else 0

# Get all PDF files in the directory, sorted numerically
pdf_files = sorted(pdf_directory.glob('*.pdf'), key=extract_number)

# Set video parameters
fps        = 5
frame_size = (2160, 2160)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video), fourcc, fps, frame_size)

# Set the bitrate (in bits per second)
bitrate = 50_000_000  # 10 Mbps
out.set(cv2.VIDEOWRITER_PROP_QUALITY, 99)  # Set quality (0-100)

for pdf_file in tqdm(pdf_files):
    # Convert PDF to image
    images = convert_from_path(str(pdf_file))
    
    # We assume each PDF has only one page
    image = images[0]
    
    # Resize image to 1080x1080
    image = image.resize(frame_size)
    
    # Convert PIL Image to numpy array for OpenCV
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Write the frame to the video
    out.write(frame)

# Release the VideoWriter
out.release()

print(f"Video saved as {output_video}")
