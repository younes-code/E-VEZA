# Caption every frame of the 3 s grid with mPLUG-Owl2.
# Run from inside the official mPLUG-Owl2 repository (folder mplug_owl2/), whose modules are imported below.
import os
import torch
from PIL import Image
from transformers import AutoTokenizer
from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from conversation import conv_templates
from model.builder import load_pretrained_model
from mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import re
import gc
import time

# ---------------------------
# Utility functions
# ---------------------------
def extract_frame_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def resize_image(image, max_size=1024):
    max_edge = max(image.size)
    if max_edge > max_size:
        ratio = max_size / float(max_edge)
        new_size = tuple([int(dim * ratio) for dim in image.size])
        image = image.resize(new_size)
    return image

def generate_caption(model, tokenizer, image_processor, img_path):
    img_dir = os.path.basename(os.path.dirname(img_path))
    img_name = os.path.basename(img_path)

    image = Image.open(img_path).convert('RGB')
    image = resize_image(image)

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    conv = conv_templates["mplug_owl2"].copy()
    inp = DEFAULT_IMAGE_TOKEN
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    caption = f"{img_dir}/{img_name} ## {generated_text}\n"

    del image_tensor, input_ids, output_ids
    torch.cuda.empty_cache()
    gc.collect()

    return caption

# ---------------------------
# Resume mechanism
# ---------------------------
def load_done_images(save_path):
    done = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for line in f:
                if "##" in line:
                    img = line.split("##")[0].strip()  # ex: Shoplifting001_x264/...jpg
                    done.add(img)
    return done

# ---------------------------
# Main processing
# ---------------------------
def main(ucf_path, save_path, skip_frames=1):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    start_time = time.time()
    processed_frames = 0

    model_path = 'MAGAer13/mplug-owl2-llama2-7b'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, load_4bit=False, device="cuda", offload_folder="offload"
    )

    done_images = load_done_images(save_path)

    for root, dirs, files in os.walk(ucf_path):
        files.sort(key=extract_frame_number)
        frame_count = 0

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)
                relative_name = os.path.basename(root) + "/" + file

                # --- RESUME CHECK ---
                if relative_name in done_images:
                    continue

                if frame_count % skip_frames == 0:
                    try:
                        print(f"Processing image: {img_path}")
                        caption = generate_caption(model, tokenizer, image_processor, img_path)
                        with open(save_path, 'a') as f:
                            f.write(caption)
                        processed_frames += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                frame_count += 1

    total_time = time.time() - start_time
    avg_time = total_time / processed_frames if processed_frames else 0
    log_msg = f"Processed {processed_frames} frames with skip_frames={skip_frames}\n" \
              f"Total time: {total_time:.2f}s, Avg per frame: {avg_time:.2f}s\n"

    print(log_msg)
    # Save timing log
    log_file_path = os.path.join(os.path.dirname(save_path), "caption_time_log.txt")
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_msg)

    torch.cuda.empty_cache()
    gc.collect()
    print("Processing complete.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Caption extracted frames with mPLUG-Owl2.")
    ap.add_argument("--frames-dir", required=True, help="folder of extracted frames (one sub-folder per video)")
    ap.add_argument("--out", default="mplug_captions.txt", help="caption file to append to")
    ap.add_argument("--skip-frames", type=int, default=1)
    args = ap.parse_args()
    main(args.frames_dir, args.out, skip_frames=args.skip_frames)

