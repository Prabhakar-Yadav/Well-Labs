#!/usr/bin/env python3
"""
Extract training patches from labeled full image
Reads LabelMe JSON and extracts patches along labeled canal regions
"""

import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import cv2

DATA_DIR = r'c:/Users/PRABHAKAR/Documents/Wells-Lab'
ANNO_V2 = os.path.join(DATA_DIR, 'annotation_v2')
FULL_IMG_DIR = os.path.join(DATA_DIR, 'annotation_v2', 'full_images')

PATCH_SIZE = 512
N_PATCHES = 50  # Extract 50 patches from labeled regions

print("="*60)
print("Extracting patches from labeled full image")
print("="*60)

# Load JSON annotation
json_path = os.path.join(FULL_IMG_DIR, 'ATTANUR_1_full.json')

if not os.path.exists(json_path):
    print(f'\nERROR: {json_path} not found!')
    print('Label the image first with LabelMe.')
    exit(1)

with open(json_path) as f:
    anno = json.load(f)

# Load metadata
meta = np.load(os.path.join(FULL_IMG_DIR, 'ATTANUR_1_meta.npy'), allow_pickle=True).item()
scale = meta['scale']
orig_W, orig_H = meta['original_size']

print(f'Found {len(anno["shapes"])} labeled shapes')
print(f'Scale factor: {scale:.3f}')

# Create mask from annotations (on downscaled coordinates)
h_down, w_down = meta['downscaled_size']
mask_down = np.zeros((h_down[1], w_down[0]), dtype=np.uint8)

canal_count = 0
fic_count = 0

for shape in anno['shapes']:
    label = shape['label'].lower().strip()
    pts = np.array(shape['points'], dtype=np.int32)

    if label in ['canal', 'channel', 'distributary']:
        # Draw thick line for canals
        if shape['shape_type'] == 'line':
            cv2.polylines(mask_down, [pts], False, 1, thickness=20)
        elif shape['shape_type'] == 'polygon':
            cv2.fillPoly(mask_down, [pts], 1)
        canal_count += 1
    elif label == 'fic':
        if shape['shape_type'] == 'line':
            cv2.polylines(mask_down, [pts], False, 2, thickness=10)
        elif shape['shape_type'] == 'polygon':
            cv2.fillPoly(mask_down, [pts], 2)
        fic_count += 1

print(f'  Canals: {canal_count}')
print(f'  FICs: {fic_count}')

if mask_down.max() == 0:
    print('\nNo labels found! Draw canals/FICs in LabelMe first.')
    exit(1)

# Find labeled regions and sample patch centers
labeled_pixels = np.where(mask_down > 0)
if len(labeled_pixels[0]) == 0:
    print('No labeled pixels found!')
    exit(1)

# Sample N_PATCHES points along labeled regions
indices = np.random.choice(len(labeled_pixels[0]),
                           min(N_PATCHES, len(labeled_pixels[0])),
                           replace=False)

patch_centers = []
for idx in indices:
    y_down = labeled_pixels[0][idx]
    x_down = labeled_pixels[1][idx]

    # Convert to original coordinates
    y_orig = int(y_down / scale)
    x_orig = int(x_down / scale)

    # Make sure patch fits
    if (x_orig > PATCH_SIZE//2 and x_orig < orig_W - PATCH_SIZE//2 and
        y_orig > PATCH_SIZE//2 and y_orig < orig_H - PATCH_SIZE//2):
        patch_centers.append((y_orig, x_orig))

print(f'\nExtracting {len(patch_centers)} patches from labeled regions...')

# Extract patches from original image
attanur1_path = os.path.join(DATA_DIR, 'TLBC_D95_ATTANUR_1_ortho.tif')

existing = len([f for f in os.listdir(ANNO_V2) if f.endswith('.jpg')])

saved = 0
with rasterio.open(attanur1_path) as src:
    for y_center, x_center in patch_centers:
        # Extract patch centered on labeled canal
        col = x_center - PATCH_SIZE // 2
        row = y_center - PATCH_SIZE // 2

        data = src.read([1, 2, 3, 4], window=Window(col, row, PATCH_SIZE, PATCH_SIZE))
        valid = data[3] == 255

        if valid.sum() < PATCH_SIZE * PATCH_SIZE * 0.7:
            continue

        rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)

        # Save patch
        patch_id = f'ATTANUR_1_labeled_{existing + saved + 1:03d}'

        np.save(os.path.join(ANNO_V2, f'{patch_id}_raw.npy'), rgb)
        Image.fromarray(rgb).save(os.path.join(ANNO_V2, f'{patch_id}.jpg'), quality=95)

        print(f'  Saved: {patch_id}')
        saved += 1

print(f'\n{"="*60}')
print(f'Success! Extracted {saved} patches from labeled canals')
print(f'Total patches in annotation_v2: {existing + saved}')
print(f'\nNext steps:')
print(f'  1. Review patches in annotation_v2/')
print(f'  2. Refine labels if needed with: python -m labelme annotation_v2')
print(f'  3. Re-train model with new data')
print(f'{"="*60}')
