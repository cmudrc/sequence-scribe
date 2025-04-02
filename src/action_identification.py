import os
import cv2
import csv
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple
import tkinter as Tk
from tkinter import filedialog

from utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
    generate_element_id,
    normalize_bbox,
    compute_iou,
    get_dominant_color,
    categorize_interactivity
)

# Set device for the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to search for the model in parent dictionaies
def find_model_path(model_filename='best.pt', search_folder='weights/icon_detect'):
    current_dir = Path(__file__).resolve().parent # Start from the script directory
    
    while current_dir != current_dir.root: # Traverse upwards till root
        model_path = current_dir / search_folder / model_filename
        if model_path.exists():
            return str(model_path) # Return the absolute path if found
        current_dir = current_dir.parent # Move one level up
        
    raise FileNotFoundError(f"Model file '{model_filename}' not found in any parent directory.")

# Initialize models globally for icon detection
ICON_DETECT_MODEL_PATH = find_model_path()
ICON_CAPTION_MODEL_NAME = 'florence2'
ICON_CAPTION_MODEL_PATH = 'Microsoft/Florence-2-base'

# Load YOLO Model for icon detection
yolo_model = get_yolo_model(ICON_DETECT_MODEL_PATH)
caption_model_processor = get_caption_model_processor(ICON_CAPTION_MODEL_NAME, ICON_CAPTION_MODEL_PATH)


def extract_frames_from_video(
    video_path: str, output_folder: str
) -> None:
    
    """
    Extract frames from a video at 1 second intervals
    """
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Accept video input
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame every second (based on FPS)
        if frame_count % fps == 0:
            frame_path = os.path.join(
                output_folder, f"frame_{saved_frame_count:04d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            print(f"Saved {frame_path}")
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Completed extracting {saved_frame_count} frames from {video_path}")


def parse_frame(frame_path: str, previous_elements: List[Dict] = None) -> List[Dict[str, Any]]:
    """
    Process an image and extract structured data about UI elements.
    """
    if previous_elements is None:
        previous_elements = []
        
    image = Image.open(frame_path)
    box_threshold = 0.05
    iou_threshold = 0.1
    use_paddleocr = False
    imgsz = 1920
    icon_process_batch_size = 64
    
    image_width, image_height = image.size
    image_np = np.array(image)
    
    # Get OCR bounding boxes
    ocr_bbox_rslt, _ = check_ocr_box(
        frame_path,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold': 0.9},
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt
    
    try:
        # Get the parsed content list
        _, _, parsed_content_list = get_som_labeled_img(
            frame_path,
            yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config={},
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            batch_size=icon_process_batch_size
        )
        
        if not parsed_content_list:
            print(f"WARNING: No elements detected in {frame_path}")
            return []
        
        # Create a structured output
        structured_data = []
        for i, element in enumerate(parsed_content_list):
            try:
                if not isinstance(element, dict):
                    print(f"WARNING: Element {i} is not a dictionary: {type(element)}")
                    continue
                
                bbox = element.get('bbox', None)
                if not isinstance(bbox, list) or len(bbox) != 4:
                    print(f"WARNING: Invalid bbox format for element {i}: {bbox}")
                    continue
                
                # Convert to pixel coordinates (for processing)
                bbox_pixels = [
                    int(bbox[0] * image_width),
                    int(bbox[1] * image_height),
                    int(bbox[2] * image_width),
                    int(bbox[3] * image_height)
                ]
                
                # Generate element ID
                element_id = generate_element_id(bbox_pixels, element.get('content', ''))
                
                # Calculate normalized bbox (cx, cy, w, h format)
                normalized_bbox = normalize_bbox(bbox_pixels, image_width, image_height)
                
                # Get dominant color
                dominant_color = get_dominant_color(image_np, bbox_pixels)
                
                # Determine interaction type
                interaction_type = categorize_interactivity(element.get('type', 'unknown'))
                
                # Get OCR confidence (if available)
                ocr_confidence = element.get('ocr_confidence', 0.0)
                if ocr_confidence is None or not isinstance(ocr_confidence, (int, float)):
                    ocr_confidence = 0.0
                
                # Compute IOU with previous elements
                max_iou = 0
                if previous_elements:
                    for prev_element in previous_elements:
                        prev_bbox_str = prev_element.get('Bounding Box', '')
                        if prev_bbox_str:
                            try:
                                # Convert string representation back to list
                                import ast
                                prev_bbox = ast.literal_eval(prev_bbox_str)
                                iou = compute_iou(bbox_pixels, prev_bbox)
                                max_iou = max(max_iou, iou)
                            except (ValueError, SyntaxError):
                                continue
                
                # Add to structured data
                structured_data.append({
                    "Image Name": os.path.basename(frame_path),
                    "Element ID": element_id,
                    "Type": element.get('type', 'unknown'),
                    "Bounding Box": bbox_pixels,
                    "Normalized Bounding Box": normalized_bbox,
                    "Interactivity": element.get('interactivity', False),
                    "Interaction Type": interaction_type,  # Changed from "Interactivity Type"
                    "Content": element.get('content', ''),
                    "OCR Confidence": ocr_confidence,
                    "IOU with Previous": max_iou,
                    "Dominant Color": dominant_color,
                })
                
            except Exception as e:
                import traceback
                print(f"ERROR processing element {i}: {str(e)}")
                print(traceback.format_exc())
                
        return structured_data
        
    except Exception as e:
        import traceback
        print(f"ERROR in get_som_labeled_img: {str(e)}")
        print(traceback.format_exc())
        raise


#def diff_frames(frame1: str, frame2: str) -> str:
#    # Apply frame differencing here
#    return "a"


def extract_sequence_from_video(
    video_path: str, 
    output_folder: str = None
) -> Tuple[List[Dict[str, Any]], str]:
    
    """
    Extract frames from video and parse each frame to detect UI elements.
    """
    
    # Create a default output folder if not specified
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(video_path), "extracted_frames")
        
    # Create frames subfolder
    frames_folder = os.path.join(output_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    # Extract frames from video
    extract_frames_from_video(video_path, frames_folder)
    
    # Parse each frame
    all_elements = []
    previous_elements = []
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Prepare CSV output file
    csv_path = os.path.join(output_folder, "parsed_elements.csv")
    headers = [
        "Image Name",
        "Element ID",
        "Type",
        "Bounding Box",
        "Normalized Bounding Box",
        "Interactivity",
        "Interaction Type",
        "Content",
        "OCR Confidence",
        "IOU with Previous",
        "Dominant Color",
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)
            try:
                print(f"Processing {frame_file}")
                parsed_data = parse_frame(frame_path, previous_elements)
                
                # Convert data for CSV output
                csv_rows = []
                for item in parsed_data:
                    row = {}
                    for key, value in item.items():
                        if key == "Bounding Box" or key == "Normalized Bounding Box":
                            row[key] = str(value)
                        elif key == "Dominant Color":
                            row[key] = str(value)
                        elif key == "OCR Confidence":
                            row[key] = float(value) if value is not None else 0.0
                        elif key == "IOU with Previous":
                            row[key] = float(value) if value is not None else 0.0
                        else:
                            row[key] = value
                    csv_rows.append(row)
                
                writer.writerows(csv_rows)
                previous_elements = parsed_data  # Update for next frame
                all_elements.extend(parsed_data)
                print(f'Successfully processed {frame_file}')
            except Exception as e:
                import traceback
                print(f"Failed to process {frame_file}: {str(e)}")
                print(traceback.format_exc())
                
    print(f"Parsed data saved to {csv_path}")
    return all_elements, csv_path


def standardize_actions(actions: list[list[str]]) -> list[list[str]]:
    # Flatten the list of actions
    flattened_actions = [action for sublist in actions for action in sublist]

    # Embed actions using a pre-trained model

    # Cluster the embedded actions

    # Use LLM to describe each cluster, providing a label for that action type

    # Map each action to its corresponding cluster label

    return actions


def extract_actions_from_videos(video_paths: list[str], output_folder: str = None) -> List[Tuple[List[Dict[str, Any]], str]]:
    """
    Process multiple videos to extract UI elements.
    """
    results = []
    for i, video_path in enumerate(video_paths):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        if output_folder:
            video_output_folder = os.path.join(output_folder, f"{video_name}")
        else:
            video_output_folder = os.path.join(os.path.dirname(video_path), f"{video_name}_data")
            
        print(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
        elements, csv_path = extract_sequence_from_video(video_path, video_output_folder)
        results.append((elements, csv_path))
        
    return results


# Function to select a video file using GUI
def select_video_file():
    root = Tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='Select video file',
        filetypes=[('Video files', '*.mp4 *.avi *.mov *.mkv')]
    )
    return file_path


# Function to select an output folder using GUI
def select_output_folder():
    root = Tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title='Select output folder')
    return folder_path


def main():
    print("Select a video file to process:")
    video_path = select_video_file()
    
    if not video_path:
        print("No video file selected. Exiting...")
        return
    
    print("Select an output folder (or cancel to use default):")
    output_folder = select_output_folder()
    
    if not output_folder:
        output_folder = os.path.join(os.path.dirname(video_path), "action_extraction_results")
        print(f"Using default output folder: {output_folder}")
        
    elements, csv_path = extract_sequence_from_video(video_path, output_folder)
    print(f"Processed {len(elements)} UI elements from the video")
    print(f"Results saved to: {csv_path}")
    

if __name__ == "__main__":
    main()


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
