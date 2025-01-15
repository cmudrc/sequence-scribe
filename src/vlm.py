import transformers
import torch


def make_vlm_model() -> transformers.Qwen2VLForConditionalGeneration:
    # Load model with memory optimizations
    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
    )

    return model


def make_vlm_processor() -> transformers.AutoProcessor:
    # Adjust pixel range for better memory performance
    min_pixels = 128 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = transformers.AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    )

    return processor
