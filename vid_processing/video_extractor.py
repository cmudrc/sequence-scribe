import cv2
import os
import tkinter as Tk
from tkinter import filedialog

def select_file():
    root = Tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title='Select video file', filetypes=[('MP4 files', '*.mp4')])
    return file_path

def select_output_folder():
    root = Tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title='Select output folder')
    return folder_path

def save_frames(video_path, output_folder):
    # Create output directory if not existing
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Accept video input
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % fps == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count//fps:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Attempting to save: {frame_path}")
            if os.path.exists(frame_path):
                print(f"Saved {frame_path}")
            else:
                print(f"Failed to save {frame_path}")
            
        frame_count += 1
        
    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()
    print("Completed extracting frames.")
        
# Use the GUI to select video file and output directory
video_path = select_file()
output_directory = select_output_folder()
if video_path and output_directory:
    save_frames(video_path, output_directory)
else:
    print("Operation Cancelled.")