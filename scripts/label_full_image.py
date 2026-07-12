#!/usr/bin/env python3
"""
Step 1: Create downscaled version of ATTANUR_1 for labeling
Step 2: Label it with LabelMe
Step 3: Extract patches from labeled regions
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import cv2

DATA_DIR = r'c:/Users/PRABHAKAR/Documents/Wells-Lab'
ANNO_V2 = os.path.join(DATA_DIR, 'annotation_v2')
FULL_IMG_DIR = os.path.join(DATA_DIR, 'annotation_v2', 'full_images')
os.makedirs(FULL_IMG_DIR, exist_ok=True)

print("="*60)
print("STEP 1: Create downscaled ATTANUR_1 for labeling")
print("="*60)

attanur1_path = os.path.join(DATA_DIR, 'TLBC_D95_ATTANUR_1_ortho.tif')

with rasterio.open(attanur1_path) as src:
    W, H = src.width, src.height
    print(f'Original size: {W}x{H}')

    # Downsample to manageable size (max 4000px on longest side)
    scale = min(4000 / max(W, H), 1.0)
    new_w = int(W * scale)
    new_h = int(H * scale)

    print(f'Downscaled size: {new_w}x{new_h} (scale={scale:.3f})')

    # Read and downsample
    data = src.read([1, 2, 3], out_shape=(3, new_h, new_w))
    rgb = data.transpose(1, 2, 0).astype(np.uint8)

    # Enhance contrast for better visibility
    for c in range(3):
        ch = rgb[:, :, c]
        lo, hi = np.percentile(ch[ch > 0], [2, 98])
        rgb[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    # Save as JPG for labeling
    output_path = os.path.join(FULL_IMG_DIR, 'ATTANUR_1_full.jpg')
    Image.fromarray(rgb).save(output_path, quality=95)

    # Save metadata for later patch extraction
    meta = {
        'scale': scale,
        'original_size': (W, H),
        'downscaled_size': (new_w, new_h)
    }
    np.save(os.path.join(FULL_IMG_DIR, 'ATTANUR_1_meta.npy'), meta)

    print(f'\nSaved: {output_path}')

print("\n" + "="*60)
print("STEP 2: Label the image")
print("="*60)
print(f'\nRun this command:')
print(f'  python -m labelme "{FULL_IMG_DIR}"')
print(f'\nIn LabelMe:')
print(f'  1. Open ATTANUR_1_full.jpg')
print(f'  2. Draw lines/polygons along canals')
print(f'  3. Label as "canal" or "fic"')
print(f'  4. Save (Ctrl+S)')
print(f'\nThen run: python extract_from_labels.py')
print("="*60)
