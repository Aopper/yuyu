import cv2
import os
import numpy as np
from ultralytics import YOLO

def v2p(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Total frames saved: {frame_count}")
    print(f"All frames saved in the folder: {output_folder}")









