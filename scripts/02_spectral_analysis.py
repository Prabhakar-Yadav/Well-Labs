"""
Stage 2 (revised) — Spectral analysis for all 5 ortho tiles
Canal knowledge:
  - Main canal + distributaries : CONCRETE  -> bright grey/white, low VARI, hard linear edges
  - FICs (Field Irrigation Ch.) : EARTHEN   -> tan/brown, low VARI, softer edges, at farm level
  - Blockages show as: sediment inside channel, vegetation encroachment (VARI spike), width anomaly

Outputs per file:
  1. RGB overview
  2. VARI map
  3. 3-class overlay  (concrete canal | FIC/earthen | vegetation)
  4. Spectral scatter  (R vs G, coloured by VARI)
  5. RGB histograms
  + combined 5-tile summary grid
"""

import os, json
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

DATA_DIR = r"c:/Users/PRABHAKAR/Documents/Wells-Lab"
OUT_DIR  = os.path.join(DATA_DIR, "outputs", "02_spectral")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    "TLBC_D95_ATTANUR_1_ortho.tif",
    "TLBC_D95_ATTANUR_2_ortho.tif",
    "TLBC_D95_ATTANUR_3_ortho.tif",
    "TLBC_D95_ATTANUR_4_ortho.tif",
    "TLBC_D95_SHAKAPUR_ortho.tif",
]

THUMB_PX    = 1500
SAMPLE_STEP = 30

VARI_CMAP = LinearSegmentedColormap.from_list(
    "vari", ["#8B4513", "#d2b48c", "#f5f5f5", "#90ee90", "#006400"])

# ── helpers ───────────────────────────────────────────────────────────────────

def vari(r, g, b):
    denom = g + r - b
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return np.clip((g - r) / denom, -1, 1)

def stretch(img_hw3, lo=2, hi=98):
    out = np.zeros_like(img_hw3, dtype=np.uint8)
    for c in range(img_hw3.shape[2]):
        ch = img_hw3[:, :, c].astype(float)
        valid = ch[ch > 0]
        if valid.size == 0: continue
        p_lo, p_hi = np.percentile(valid, [lo, hi])
        if p_hi > p_lo:
            out[:, :, c] = np.clip((ch-p_lo)/(p_hi-p_lo)*255, 0, 255).astype(np.uint8)
    return out

def read_overview(fpath, target_px=THUMB_PX):
    with rasterio.open(fpath) as src:
        W, H  = src.width, src.height
        scale = max(1, max(W, H) // target_px)
        ow, oh = W // scale, H // scale
        data = src.read([1,2,3,4], out_shape=(4, oh, ow)).astype(np.float32)
    return data, scale   # shape (4, oh, ow)

def sample_pixels(fpath, grid=(5,4), patch=512, step=SAMPLE_STEP):
    """Grid-sample patches, return flat r,g,b,vari arrays."""
    all_r, all_g, all_b, all_v = [], [], [], []
    with rasterio.open(fpath) as src:
        W, H = src.width, src.height
        for ri in range(grid[0]):
            for ci in range(grid[1]):
                co = int(ci * W/grid[1] + (W/grid[1]-patch)/2)
                ro = int(ri * H/grid[0] + (H/grid[0]-patch)/2)
                co = max(0, min(co, W-patch))
                ro = max(0, min(ro, H-patch))
                from rasterio.windows import Window
                win  = Window(co, ro, patch, patch)
                data = src.read([1,2,3,4], window=win).astype(np.float32)
                a    = data[3]
                mask = a > 128
                if mask.sum() < 500: continue
                r,g,b = data[0][mask], data[1][mask], data[2][mask]
                v     = vari(data[0], data[1], data[2])[mask]
                all_r.append(r[::step]); all_g.append(g[::step])
                all_b.append(b[::step]); all_v.append(v[::step])
    return (np.concatenate(x) for x in [all_r, all_g, all_b, all_v])

# ── canal class masks (applied to overview arrays) ────────────────────────────
# CONCRETE canal: bright, grey (R~G~B), low VARI
# FIC / earthen : moderate brightness, warmer tone (R>B), low-mid VARI
# Vegetation    : green, high VARI

def classify(r, g, b, alpha):
    valid  = alpha > 128
    bright = (r + g + b) / 3.0
    grey   = np.abs(r.astype(float) - g) + np.abs(g.astype(float) - b)  # low = grey
    vi     = vari(r, g, b)

    # Concrete canal: grey + bright + low VARI
    concrete = valid & (vi < 0.05) & (bright > 130) & (grey < 25)

    # FIC / earthen: warm (R>B), moderate brightness, low VARI, NOT grey
    earthen  = valid & (vi < 0.08) & (bright > 80) & (bright < 200) \
                     & (r > b + 5) & (grey >= 15)

    # Vegetation: high VARI
    veg      = valid & (vi > 0.12)

    return concrete, earthen, veg, valid

# ── per-file analysis loop ─────────────────────────────────────────────────────

all_stats = {}

for fname in FILES:
    tag   = fname.replace("_ortho.tif", "")
    fpath = os.path.join(DATA_DIR, fname)
    size_gb = os.path.getsize(fpath) / 1e9
    print(f"\n{'='*60}\n  {tag}  ({size_gb:.1f} GB)\n{'='*60}")

    # --- overview arrays ---
    data, scale = read_overview(fpath)
    r, g, b, a  = data[0], data[1], data[2], data[3]
    rgb_s = stretch(np.transpose(data[:3], (1,2,0)).astype(np.uint8))
    vi    = vari(r, g, b)
    vi[a == 0] = np.nan

    concrete, earthen, veg, valid = classify(r, g, b, a)

    def pct(m): return m.sum() / valid.sum() * 100 if valid.sum() else 0

    stats = {
        "file": fname, "size_gb": round(size_gb, 2),
        "scale": scale,
        "concrete_canal_pct": round(pct(concrete), 2),
        "earthen_fic_pct":    round(pct(earthen),  2),
        "vegetation_pct":     round(pct(veg),       2),
    }

    # --- sampled pixels ---
    print("  Sampling pixels ...")
    sr, sg, sb, sv = sample_pixels(fpath)
    sr = np.array(list(sr)); sg = np.array(list(sg))
    sb = np.array(list(sb)); sv = np.array(list(sv))
    stats["sampled_px"] = int(len(sr))
    stats["vari_mean"]  = round(float(sv.mean()), 3)
    stats["vari_std"]   = round(float(sv.std()),  3)
    stats["R_median"]   = round(float(np.median(sr)), 1)
    stats["G_median"]   = round(float(np.median(sg)), 1)
    stats["B_median"]   = round(float(np.median(sb)), 1)

    print(f"  Concrete canal: {stats['concrete_canal_pct']}%  |  "
          f"FIC/earthen: {stats['earthen_fic_pct']}%  |  "
          f"Vegetation: {stats['vegetation_pct']}%")

    all_stats[tag] = stats

    # ── figure: 5-panel per file ──────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 9))
    fig.suptitle(
        f"{tag}   ({size_gb:.1f} GB | scale 1:{scale})\n"
        f"Concrete canal: {stats['concrete_canal_pct']}%  |  "
        f"FIC/earthen: {stats['earthen_fic_pct']}%  |  "
        f"Vegetation: {stats['vegetation_pct']}%",
        fontsize=11)

    gs = fig.add_gridspec(2, 5, hspace=0.35, wspace=0.25)

    # Panel 1: RGB
    ax1 = fig.add_subplot(gs[:, 0:2])
    ax1.imshow(rgb_s)
    ax1.set_title("RGB (2-98% stretch)", fontsize=9)
    ax1.axis("off")

    # Panel 2: VARI map
    ax2 = fig.add_subplot(gs[:, 2])
    im = ax2.imshow(vi, cmap=VARI_CMAP, vmin=-0.3, vmax=0.5)
    ax2.set_title("VARI index\nBrown=bare/concrete  Green=veg", fontsize=8)
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.05, pad=0.02)

    # Panel 3: 3-class overlay
    ax3 = fig.add_subplot(gs[:, 3])
    seg = np.zeros((*vi.shape, 4), dtype=np.uint8)
    seg[concrete] = [200, 200, 200, 200]   # grey  — concrete canal
    seg[earthen]  = [180, 120,  60, 200]   # brown — FIC/earthen
    seg[veg]      = [ 34, 139,  34, 160]   # green — vegetation
    ax3.imshow(rgb_s)
    ax3.imshow(seg)
    legend = [Patch(facecolor="#c8c8c8", label="Concrete canal"),
              Patch(facecolor="#b4783c", label="FIC / earthen"),
              Patch(facecolor="#228b22", label="Vegetation")]
    ax3.legend(handles=legend, loc="lower right", fontsize=6)
    ax3.set_title("3-class overlay\n(concrete / FIC / veg)", fontsize=8)
    ax3.axis("off")

    # Panel 4 top: RGB histograms
    ax4 = fig.add_subplot(gs[0, 4])
    for ch, col, lbl in zip([sr, sg, sb],
                             ["#e05c5c","#4eae4e","#4e7aae"],
                             ["R","G","B"]):
        ax4.hist(ch, bins=80, color=col, alpha=0.6, label=lbl, density=True)
    ax4.set_title("RGB histogram\n(sampled)", fontsize=8)
    ax4.set_xlabel("Value"); ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    # Panel 4 bottom: R vs G scatter coloured by VARI
    ax5 = fig.add_subplot(gs[1, 4])
    idx = np.random.choice(len(sr), min(30000, len(sr)), replace=False)
    sc  = ax5.scatter(sr[idx], sg[idx], c=sv[idx], cmap=VARI_CMAP,
                      vmin=-0.3, vmax=0.5, s=0.5, alpha=0.4)
    ax5.set_xlabel("Red"); ax5.set_ylabel("Green")
    ax5.set_title("R vs G (VARI colour)", fontsize=8)
    ax5.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax5, fraction=0.08, pad=0.02, label="VARI")

    out_png = os.path.join(OUT_DIR, f"{tag}_spectral.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_png}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "spectral_stats_all.json"), "w") as f:
    json.dump(all_stats, f, indent=2)
print("\nAll stats saved -> outputs/02_spectral/spectral_stats_all.json")

# ── Combined 5-tile summary ────────────────────────────────────────────────────
print("\nBuilding combined summary grid ...")
fig, axes = plt.subplots(3, 5, figsize=(26, 14))
fig.suptitle("Stage 2 Summary — All 5 Tiles: RGB | VARI | 3-class", fontsize=13)

for col_i, fname in enumerate(FILES):
    tag   = fname.replace("_ortho.tif", "")
    fpath = os.path.join(DATA_DIR, fname)
    data, scale = read_overview(fpath, target_px=600)
    r, g, b, a  = data[0], data[1], data[2], data[3]
    rgb_s = stretch(np.transpose(data[:3], (1,2,0)).astype(np.uint8))
    vi    = vari(r, g, b); vi[a==0] = np.nan
    concrete, earthen, veg, valid = classify(r, g, b, a)

    seg = np.zeros((*vi.shape, 4), dtype=np.uint8)
    seg[concrete] = [200,200,200,200]
    seg[earthen]  = [180,120, 60,200]
    seg[veg]      = [ 34,139, 34,160]

    s = all_stats[tag]
    title_short = tag.replace("TLBC_D95_","")

    axes[0, col_i].imshow(rgb_s)
    axes[0, col_i].set_title(f"{title_short}\n{s['size_gb']} GB", fontsize=8)
    axes[0, col_i].axis("off")

    axes[1, col_i].imshow(vi, cmap=VARI_CMAP, vmin=-0.3, vmax=0.5)
    axes[1, col_i].set_title(f"VARI  mean={s['vari_mean']}", fontsize=8)
    axes[1, col_i].axis("off")

    axes[2, col_i].imshow(rgb_s)
    axes[2, col_i].imshow(seg)
    axes[2, col_i].set_title(
        f"Concrete:{s['concrete_canal_pct']}%  "
        f"FIC:{s['earthen_fic_pct']}%\n"
        f"Veg:{s['vegetation_pct']}%", fontsize=7)
    axes[2, col_i].axis("off")

# row labels
for row, lbl in enumerate(["RGB", "VARI", "3-class"]):
    axes[row, 0].set_ylabel(lbl, fontsize=10, labelpad=4)

legend = [Patch(facecolor="#c8c8c8", label="Concrete canal"),
          Patch(facecolor="#b4783c", label="FIC / earthen"),
          Patch(facecolor="#228b22", label="Vegetation")]
fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0, 0.04, 1, 1])
combined = os.path.join(OUT_DIR, "ALL_TILES_stage2_summary.png")
plt.savefig(combined, dpi=120, bbox_inches="tight")
plt.close()
print(f"Combined summary -> {combined}")
print("\nStage 2 complete.")
