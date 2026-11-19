"""
The script generates captions using the Hugging Face BLIP-2 model, and saves the captions to a text file.

"""
import os
import re
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import time
import gc

def extract_frame_number(filename):
    """
    Extracts the numerical part from a filename.

    Args:
        filename (str): The input filename.

    Returns:
        int: Extracted numerical part or float('inf') if not found.
    """
    # Extract numerical part from the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def generate_caption(model, processor, img_path, device):
    """
    Generates a caption for the given image using the provided model and processor.

    Args:
        model (Blip2ForConditionalGeneration): The BLIP-2 model for caption generation.
        processor (Blip2Processor): The BLIP-2 processor for image preprocessing.
        img_path (str): The path to the input image.
        device (str): The device to use for inference (e.g., "cuda" or "cpu").

    Returns:
        str: The generated caption in the format "directory/image_name ## generated_text".
    """
    # Extract the image name without the path
    img_name = os.path.basename(img_path)

    raw_image = Image.open(img_path).convert('RGB')

    # Preprocess the image
    inputs = processor(raw_image, return_tensors="pt").to(device)

    # print(f"Before generating caption - Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB, Cached: {torch.cuda.memory_cached(device) / 1e9:.2f} GB")

    # Generate the caption
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=1000, num_beams=5)

    # print(f"After generating caption - Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB, Cached: {torch.cuda.memory_cached(device) / 1e9:.2f} GB")

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Explicitly release PyTorch tensors
    inputs = None
    generated_ids = None
    torch.cuda.empty_cache()

    # Run garbage collector to free up more memory
    gc.collect()

    result = f'{os.path.basename(os.path.dirname(img_path))}/{img_name}'
    print(result)
    return f'{os.path.basename(os.path.dirname(img_path))}/{img_name} ## {generated_text}\n'


def main(ucf_path, save_path, skip_frames=90):
    """
    Main function to process frames, generate captions, and save them to a text file.

    Args:
        ucf_path (str): The path to the UCF dataset directory.
        save_path (str): The path to the output captions text file.
        skip_frames (int): Number of frames to skip between generating captions 90 by default.

    Returns:
        None
    """
    # Set up device to use (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Hugging Face BLIP-2 processor and model with float16 precision
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

    # Loop through all subdirectories in the UCF directory
    for root, dirs, files in os.walk(ucf_path):
        # Sort the files based on extracted numerical parts
        files.sort(key=extract_frame_number)

        frame_count = 0
        generate_caption_flag = True

        for file in files:
            start_total_time = time.time()
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(root, file)

                if generate_caption_flag:
                    # Generate caption for the current image
                    caption = generate_caption(model, processor, img_path, device)

                    # Save the caption to a text file
                    with open(save_path, 'a') as file:
                        file.write(caption)

                    generate_caption_flag = False

                    # Free GPU memory after each caption is generated and saved
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()

                # Increment the frame counter
                frame_count += 1

                # If the specified number of frames have been processed, reset the counter
                if frame_count >= skip_frames:
                    frame_count = 0
                    generate_caption_flag = True

            end_total_time = time.time()
            # print(f'Time for generating one caption: {end_total_time - start_total_time} seconds')

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Caption extracted frames with BLIP-2 (version used for the released captions).")
    ap.add_argument("--frames-dir", required=True, help="folder of extracted frames (one sub-folder per video)")
    ap.add_argument("--out", default="blip_captions.txt", help="caption file to append to")
    ap.add_argument("--skip-frames", type=int, default=90)
    args = ap.parse_args()
    main(args.frames_dir, args.out, skip_frames=args.skip_frames)
