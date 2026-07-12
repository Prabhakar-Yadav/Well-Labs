#!/usr/bin/env python3
"""
Extract patches with better spacing and canal-likely targeting
Version 2: Larger stride, focus on areas with linear features
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import cv2

DATA_DIR = r'c:/Users/PRABHAKAR/Documents/Wells-Lab'
ANNO_V2 = os.path.join(DATA_DIR, 'annotation_v2')
os.makedirs(ANNO_V2, exist_ok=True)

PATCH_SIZE = 512
N_PATCHES_PER_IMAGE = 50
STRIDE = 1024  # Larger spacing between patches (was random before)

def has_linear_features(rgb):
    """Check if patch has linear features (potential canals)"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Look for lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=100, maxLineGap=20)

    if lines is not None and len(lines) > 2:
        return True, len(lines)
    return False, 0

# Check existing patches
existing_manual = [f for f in os.listdir(ANNO_V2) if '_manual_' in f and f.endswith('.jpg')]
existing_count = len(existing_manual)
print(f'Existing manual patches: {existing_count}')

# ============================================================================
# ATTANUR_1 - Left region, with linear feature detection
# ============================================================================

print('\n' + '='*60)
print('ATTANUR_1 - Left region with canal detection')
print('='*60)

attanur1_path = os.path.join(DATA_DIR, 'TLBC_D95_ATTANUR_1_ortho.tif')

candidates_a1 = []

with rasterio.open(attanur1_path) as src:
    W1, H1 = src.width, src.height
    left_width = int(W1 * 0.5)  # Increased to 50% for more coverage

    print(f'Scanning with stride={STRIDE}px...')

    # Grid scan with stride
    for row in range(0, H1 - PATCH_SIZE, STRIDE):
        for col in range(0, max(1, left_width - PATCH_SIZE), STRIDE):
            data = src.read([1, 2, 3, 4], window=Window(col, row, PATCH_SIZE, PATCH_SIZE))
            valid = data[3] == 255

            if valid.sum() < PATCH_SIZE * PATCH_SIZE * 0.7:
                continue

            rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)

            # Check for linear features
            has_lines, line_count = has_linear_features(rgb)

            candidates_a1.append({
                'rgb': rgb,
                'row': row,
                'col': col,
                'has_lines': has_lines,
                'line_count': line_count
            })

    print(f'Found {len(candidates_a1)} candidates')

# Sort by line count (most likely to have canals first)
candidates_a1.sort(key=lambda x: x['line_count'], reverse=True)

print(f'\nSaving top {N_PATCHES_PER_IMAGE} patches with most linear features...')
saved_a1 = 0

for candidate in candidates_a1[:N_PATCHES_PER_IMAGE]:
    patch_id = f'ATTANUR_1_v2_{existing_count + saved_a1 + 1:03d}'

    np.save(os.path.join(ANNO_V2, f'{patch_id}_raw.npy'), candidate['rgb'])
    Image.fromarray(candidate['rgb']).save(os.path.join(ANNO_V2, f'{patch_id}.jpg'), quality=95)

    status = 'LINES' if candidate['has_lines'] else 'plain'
    print(f'  {patch_id}: {candidate["line_count"]} lines ({status})')
    saved_a1 += 1

# ============================================================================
# ATTANUR_4 - Center region
# ============================================================================

print('\n' + '='*60)
print('ATTANUR_4 - Center region with canal detection')
print('='*60)

attanur4_path = os.path.join(DATA_DIR, 'TLBC_D95_ATTANUR_4_ortho.tif')

candidates_a4 = []

with rasterio.open(attanur4_path) as src:
    W4, H4 = src.width, src.height
    center_start = int(W4 * 0.25)
    center_end = int(W4 * 0.75)

    print(f'Scanning with stride={STRIDE}px...')

    for row in range(0, H4 - PATCH_SIZE, STRIDE):
        for col in range(center_start, max(center_start+1, center_end - PATCH_SIZE), STRIDE):
            data = src.read([1, 2, 3, 4], window=Window(col, row, PATCH_SIZE, PATCH_SIZE))
            valid = data[3] == 255

            if valid.sum() < PATCH_SIZE * PATCH_SIZE * 0.7:
                continue

            rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)
            has_lines, line_count = has_linear_features(rgb)

            candidates_a4.append({
                'rgb': rgb,
                'row': row,
                'col': col,
                'has_lines': has_lines,
                'line_count': line_count
            })

    print(f'Found {len(candidates_a4)} candidates')

candidates_a4.sort(key=lambda x: x['line_count'], reverse=True)

print(f'\nSaving top {N_PATCHES_PER_IMAGE} patches with most linear features...')
saved_a4 = 0

for candidate in candidates_a4[:N_PATCHES_PER_IMAGE]:
    patch_id = f'ATTANUR_4_v2_{existing_count + saved_a1 + saved_a4 + 1:03d}'

    np.save(os.path.join(ANNO_V2, f'{patch_id}_raw.npy'), candidate['rgb'])
    Image.fromarray(candidate['rgb']).save(os.path.join(ANNO_V2, f'{patch_id}.jpg'), quality=95)

    status = 'LINES' if candidate['has_lines'] else 'plain'
    print(f'  {patch_id}: {candidate["line_count"]} lines ({status})')
    saved_a4 += 1

# ============================================================================
# Summary
# ============================================================================

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'New patches extracted: {saved_a1 + saved_a4}')
print(f'  ATTANUR_1: {saved_a1} (prioritized by linear features)')
print(f'  ATTANUR_4: {saved_a4} (prioritized by linear features)')
print(f'Total patches in annotation_v2: {existing_count + saved_a1 + saved_a4}')
print(f'\nPatches are sorted by likelihood of containing canals')
print(f'First ~20 patches most likely have canals!')
print(f'\nNext: python -m labelme annotation_v2')
print('='*60)
