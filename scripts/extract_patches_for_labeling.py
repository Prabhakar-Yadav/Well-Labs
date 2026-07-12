#!/usr/bin/env python3
"""
Extract patches from ATTANUR_1 (left region) and ATTANUR_4 (center region)
for manual labeling with LabelMe.

Saves to annotation_v2/ directory:
  - *_raw.npy (for training)
  - *.jpg (for LabelMe annotation)

After running this, use LabelMe to annotate the JPG files:
  labelme annotation_v2/
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

DATA_DIR = r'c:/Users/PRABHAKAR/Documents/Wells-Lab'
ANNO_V2 = os.path.join(DATA_DIR, 'annotation_v2')
os.makedirs(ANNO_V2, exist_ok=True)

PATCH_SIZE = 512
N_PATCHES_PER_IMAGE = 50

# Check how many patches already exist
existing = len([f for f in os.listdir(ANNO_V2) if f.endswith('.jpg')])
print(f'Existing patches in annotation_v2: {existing}')

# ============================================================================
# Extract from ATTANUR_1 (left 40% region)
# ============================================================================

print('\n' + '='*60)
print('Extracting from ATTANUR_1 (left region)...')
print('='*60)

attanur1_path = os.path.join(DATA_DIR, 'TLBC_D95_ATTANUR_1_ortho.tif')

with rasterio.open(attanur1_path) as src:
    W1, H1 = src.width, src.height
    print(f'Image size: {W1}x{H1}')

    left_region_width = int(W1 * 0.4)
    print(f'Left region: 0 to {left_region_width} (40% of width)')

    patches_a1 = []
    attempts = 0
    max_attempts = N_PATCHES_PER_IMAGE * 10

    while len(patches_a1) < N_PATCHES_PER_IMAGE and attempts < max_attempts:
        col = np.random.randint(0, max(1, left_region_width - PATCH_SIZE))
        row = np.random.randint(0, max(1, H1 - PATCH_SIZE))

        data = src.read([1, 2, 3, 4], window=Window(col, row, PATCH_SIZE, PATCH_SIZE))
        valid = data[3] == 255

        if valid.sum() > PATCH_SIZE * PATCH_SIZE * 0.7:
            rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)
            patches_a1.append({
                'rgb': rgb,
                'source': 'ATTANUR_1',
                'row': row,
                'col': col
            })

            # Save immediately
            patch_id = f'ATTANUR_1_manual_{len(patches_a1):03d}'

            # Save raw.npy for training
            np.save(os.path.join(ANNO_V2, f'{patch_id}_raw.npy'), rgb)

            # Save JPG for LabelMe
            Image.fromarray(rgb).save(os.path.join(ANNO_V2, f'{patch_id}.jpg'), quality=95)

            print(f'  Saved: {patch_id}')

        attempts += 1

print(f'\nExtracted {len(patches_a1)} patches from ATTANUR_1')

# ============================================================================
# Extract from ATTANUR_4 (center 60% region)
# ============================================================================

print('\n' + '='*60)
print('Extracting from ATTANUR_4 (center region)...')
print('='*60)

attanur4_path = os.path.join(DATA_DIR, 'TLBC_D95_ATTANUR_4_ortho.tif')

with rasterio.open(attanur4_path) as src:
    W4, H4 = src.width, src.height
    print(f'Image size: {W4}x{H4}')

    center_start = int(W4 * 0.2)
    center_end = int(W4 * 0.8)
    print(f'Center region: {center_start} to {center_end} (middle 60%)')

    patches_a4 = []
    attempts = 0
    max_attempts = N_PATCHES_PER_IMAGE * 10

    while len(patches_a4) < N_PATCHES_PER_IMAGE and attempts < max_attempts:
        col = np.random.randint(center_start, max(center_start+1, center_end - PATCH_SIZE))
        row = np.random.randint(0, max(1, H4 - PATCH_SIZE))

        data = src.read([1, 2, 3, 4], window=Window(col, row, PATCH_SIZE, PATCH_SIZE))
        valid = data[3] == 255

        if valid.sum() > PATCH_SIZE * PATCH_SIZE * 0.7:
            rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)
            patches_a4.append({
                'rgb': rgb,
                'source': 'ATTANUR_4',
                'row': row,
                'col': col
            })

            # Save immediately
            patch_id = f'ATTANUR_4_manual_{len(patches_a4):03d}'

            # Save raw.npy for training
            np.save(os.path.join(ANNO_V2, f'{patch_id}_raw.npy'), rgb)

            # Save JPG for LabelMe
            Image.fromarray(rgb).save(os.path.join(ANNO_V2, f'{patch_id}.jpg'), quality=95)

            print(f'  Saved: {patch_id}')

        attempts += 1

print(f'\nExtracted {len(patches_a4)} patches from ATTANUR_4')

# ============================================================================
# Summary
# ============================================================================

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'Total patches extracted: {len(patches_a1) + len(patches_a4)}')
print(f'  ATTANUR_1 (left): {len(patches_a1)}')
print(f'  ATTANUR_4 (center): {len(patches_a4)}')
print(f'\nSaved to: {ANNO_V2}')
print(f'\nNext steps:')
print(f'  1. cd "{ANNO_V2}"')
print(f'  2. labelme')
print(f'  3. Label canals, FICs, or "not canal"')
print(f'  4. JSON files will be saved automatically')
print(f'  5. Re-train with updated annotations')
print('='*60)
