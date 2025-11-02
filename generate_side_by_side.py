import os
from PIL import Image
import numpy as np

# ========== CONFIG ==========
IMAGE_DIR = "images"
MASK_DIR = "combined_masks"
PRED_DIR = "predictions"
SIDE_DIR = "side_by_side"

os.makedirs(SIDE_DIR, exist_ok=True)

# ========== FILES ==========
def get_file_dict(directory, extensions=(".png", ".jpg", ".jpeg")):
    files = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
    return {os.path.splitext(f)[0]: os.path.join(directory, f) for f in files}

image_dict = get_file_dict(IMAGE_DIR)
mask_dict = get_file_dict(MASK_DIR)
pred_dict = get_file_dict(PRED_DIR)

common_keys = list(set(image_dict.keys()) & set(mask_dict.keys()) & set(pred_dict.keys()))
common_keys.sort()

print(f"Found {len(common_keys)} images for side-by-side visualization.")

# ========== GENERATE COMBINED ==========
for key in common_keys:
    img_path = image_dict[key]
    mask_path = mask_dict[key]
    pred_path = pred_dict[key]

    img = Image.open(img_path).convert("RGB").resize((512, 512))
    
    # Ground truth mask as RGB
    gt_mask = np.array(Image.open(mask_path).resize((512, 512)))
    gt_color = np.zeros((512, 512, 3), dtype=np.uint8)
    gt_color[gt_mask == 255] = [255, 255, 255]
    gt_color[gt_mask == 127] = [127, 127, 127]
    gt_mask_img = Image.fromarray(gt_color)

    # Predicted mask as RGB
    pred_mask = np.array(Image.open(pred_path).resize((512, 512)))
    pred_color = np.zeros((512, 512, 3), dtype=np.uint8)
    pred_color[pred_mask == 255] = [255, 255, 255]
    pred_color[pred_mask == 127] = [127, 127, 127]
    pred_mask_img = Image.fromarray(pred_color)

    # Combine side by side
    combined = Image.new("RGB", (512*3, 512))
    combined.paste(img, (0, 0))
    combined.paste(gt_mask_img, (512, 0))
    combined.paste(pred_mask_img, (512*2, 0))

    combined.save(os.path.join(SIDE_DIR, f"{key}.png"))

print(f"✅ Done! Saved combined images to '{SIDE_DIR}' folder.")
