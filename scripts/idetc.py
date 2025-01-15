import src as scribe

video_path = "/content/drive/MyDrive/2025 ORNL CMU Collaboration/videos/demo/Test Recording-20241211_111116-Meeting Recording.mp4"  # Replace with your video path

# Extract sequence of actions
actions = scribe.extract_sequence(video_path)

# Normalize
