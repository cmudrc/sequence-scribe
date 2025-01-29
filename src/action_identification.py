import os
import cv2


def extract_frames_from_video(
    video_path: str, output_folder: str, subsample_rate: int = 1
) -> None:
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return

    frame_count = 0  # Initialize the overall frame count
    saved_frame_count = 0  # Initialize a counter for saved frames
    while True:
        ret, frame = video.read()

        # Break the loop if no more frames are available
        if not ret:
            break

        # Save the frame only if it's a multiple of the subsample rate
        if frame_count % subsample_rate == 0:
            frame_filename = os.path.join(
                output_folder, f"frame_{saved_frame_count}.png"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    video.release()


def parse_frame(frame_path: str) -> str:
    # Apply omniparser here
    return "0"


def diff_frames(frame1: str, frame2: str) -> str:
    # Apply frame differencing here
    return "a"


def extract_sequence_from_video(video_path) -> list[str]:
    # Extract frames from video
    frames_folder = "frames"
    extract_frames_from_video(video_path, frames_folder)

    # Parse each frame and diff subsequent frames
    actions = []
    frame_files = sorted(os.listdir(frames_folder))
    for i in range(len(frame_files) - 1):
        frame1 = parse_frame(os.path.join(frames_folder, frame_files[i]))
        frame2 = parse_frame(os.path.join(frames_folder, frame_files[i + 1]))
        action = diff_frames(frame1, frame2)
        actions.append(action)

    return actions


def standardize_actions(actions: list[list[str]]) -> list[list[str]]:
    # Flatten the list of actions
    flattened_actions = [action for sublist in actions for action in sublist]

    # Embed actions using a pre-trained model

    # Cluster the embedded actions

    # Use LLM to describe each cluster, providing a label for that action type

    # Map each action to its corresponding cluster label

    return actions


def extract_actions_from_videos(video_paths: list[str]) -> list[list[str]]:
    actions = []
    for video_path in video_paths:
        actions.append(extract_sequence_from_video(video_path))
    return standardize_actions(actions)


#
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import torch.amp
# from .vlm import make_vlm_processor, make_vlm_model
#
#
# def extract_action(path_to_images: str, start_index: int, end_index: int) -> str:
#
#     PROMPT = """
#     You are analyzing a sequence of images depicting an atomic action on a graphic user interface. The sequence contains:
#
#     1. Two full-interface images (before and after the action), with the cursor's position annotated inside a green bounding box for reference (note: the bounding box is for annotation purposes and not part of the interface).
#     2. Two cropped, detailed images focused on the cursor's immediate surroundings (before and after the action).
#
#     Your task is to identify and describe the cursor's action based on these images. Include details such as the object clicked (e.g., button, menu item) and its context. For drags, elaborate on the dragged item, its starting and ending positions, and the intent or effect of the drag.
#
#     Return your response as a JSON dictionary with the following keys:
#     - `action_name`: A concise label for the action (e.g., "click_menu", "drag_slider").
#     - `action_description`: A detailed explanation of the action, including all relevant specifics.
#
#     Ensure the description is clear, complete, and focuses only on the information visible in the images. The output will be used directly in downstream processes, so ONLY return the JSON dictionary.
#     """
#
#     # Messages to send
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": f"file://{path_to_images}/{start_index}_prompt.png",
#                 },
#                 {
#                     "type": "image",
#                     "image": f"file://{path_to_images}/{end_index}_prompt.png",
#                 },
#                 {
#                     "type": "image",
#                     "image": f"file://{path_to_images}/{start_index}_crop.png",
#                 },
#                 {
#                     "type": "image",
#                     "image": f"file://{path_to_images}/{end_index}_crop.png",
#                 },
#                 {"type": "text", "text": PROMPT},
#             ],
#         }
#     ]
#
#     processor = make_vlm_processor()
#     model = make_vlm_model()
#
#     # Prepare inputs
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda", non_blocking=True)
#
#     # Perform inference with mixed precision
#     with torch.amp.autocast("cuda"):
#         generated_ids = model.generate(**inputs, max_new_tokens=64)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :]
#         for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )
#
#     return output_text[0]
