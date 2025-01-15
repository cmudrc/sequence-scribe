from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.amp
from .vlm import make_vlm_processor, make_vlm_model


def extract_action(path_to_images: str, start_index: int, end_index: int) -> str:

    PROMPT = """
    You are analyzing a sequence of images depicting an atomic action on a graphic user interface. The sequence contains:

    1. Two full-interface images (before and after the action), with the cursor's position annotated inside a green bounding box for reference (note: the bounding box is for annotation purposes and not part of the interface).
    2. Two cropped, detailed images focused on the cursor's immediate surroundings (before and after the action).

    Your task is to identify and describe the cursor's action based on these images. Include details such as the object clicked (e.g., button, menu item) and its context. For drags, elaborate on the dragged item, its starting and ending positions, and the intent or effect of the drag.

    Return your response as a JSON dictionary with the following keys:
    - `action_name`: A concise label for the action (e.g., "click_menu", "drag_slider").
    - `action_description`: A detailed explanation of the action, including all relevant specifics.

    Ensure the description is clear, complete, and focuses only on the information visible in the images. The output will be used directly in downstream processes, so ONLY return the JSON dictionary.
    """

    # Messages to send
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{path_to_images}/{start_index}_prompt.png",
                },
                {
                    "type": "image",
                    "image": f"file://{path_to_images}/{end_index}_prompt.png",
                },
                {
                    "type": "image",
                    "image": f"file://{path_to_images}/{start_index}_crop.png",
                },
                {
                    "type": "image",
                    "image": f"file://{path_to_images}/{end_index}_crop.png",
                },
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    processor = make_vlm_processor()
    model = make_vlm_model()

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda", non_blocking=True)

    # Perform inference with mixed precision
    with torch.amp.autocast("cuda"):
        generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


def extract_sequence(video_path) -> list[str]:
    return ["0"]


def standardize_actions(actions: list[str]) -> list[str]:
    return ["0"]
