import os
import cv2

def break_video_into_frames(video_path, output_folder, frames_per_group=10, subsample_rate=1):
    """Breaks a video into frames and saves them in groups, with subsampling.

    Args:
        video_path: Path to the video file.
        output_folder: The folder to save the frames.
        frames_per_group: Number of frames to group together.
        subsample_rate: The rate at which to subsample frames (e.g., 2 to skip every other frame).
    """

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return

    frame_count = 0  # Initialize the overall frame count
    group_count = 0
    saved_frame_count = 0  # Initialize a counter for saved frames
    while True:
        ret, frame = video.read()

        # Break the loop if no more frames are available
        if not ret:
            break

        # Save the frame only if it's a multiple of the subsample rate
        if frame_count % subsample_rate == 0:
            # Zero-pad the group_count to ensure proper sorting
            group_folder = os.path.join(output_folder, f"group_{group_count:04d}")
            os.makedirs(group_folder, exist_ok=True)

            # Reset frame count for each group
            if saved_frame_count % frames_per_group == 0:
                group_frame_count = 0

            frame_filename = os.path.join(group_folder, f"frame_{group_frame_count}.png")  # Zero-pad frame filenames
            cv2.imwrite(frame_filename, frame)

            saved_frame_count += 1
            group_frame_count += 1  # Increment frame count within the group

            if saved_frame_count % frames_per_group == 0:
                group_count += 1

        frame_count += 1

    # Release the video capture object
    video.release()
    print(f"Video successfully broken into frames and saved in groups of {frames_per_group} in {output_folder}, with subsampling rate {subsample_rate}")
