from pathlib import Path
import cv2
import mediapipe as mp
from tqdm import tqdm
from PIL import Image
import numpy as np

# === DIRECTORIES ===
DATA_ROOT = Path("/app/data/img_align_celeba")
input_dir  = DATA_ROOT / "original"
output_dir = DATA_ROOT / "clean"

output_dir.mkdir(parents=True, exist_ok=True)

# === Initialize Mediapipe ===
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def remove_background(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = mp_selfie_segmentation.process(img_rgb)

    mask = result.segmentation_mask > 0.1
    output = img_bgr.copy()
    output[~mask] = (255, 255, 255)
    return output

# === Process ===
image_paths = list(input_dir.glob("*.jpg"))
print(f"Found {len(image_paths)} images to clean.")

for img_path in tqdm(image_paths):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Error reading {img_path}")
        continue

    img_clean = remove_background(img_bgr)

    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), img_clean)

print("Background cleaning completed!")
