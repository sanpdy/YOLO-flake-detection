from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Ground-truth Monark val ─────────────────────────
IMG_VAL = Path('/home/sankalp/yolo_flake_detection/monark_data/images/val')
LBL_VAL = Path('/home/sankalp/yolo_flake_detection/monark_data/labels/val')

# ─── Your model’s outputs ───────────────────────────
PRED_DIR     = Path('/home/sankalp/yolo_flake_detection/inference_segmentation_results/seg_outputs')
PRED_IMG_DIR = PRED_DIR                       # e.g. 2-080.jpg, 21-099.jpg, etc.
PRED_LBL_DIR = PRED_DIR / 'labels'            # if you ever want to draw the predicted polygons too

# ─── Where to dump the comparison plots ────────────
OUT_DIR = Path('gt_pred_segmentation')
OUT_DIR.mkdir(exist_ok=True)

def draw_gt_seg(img_path, lbl_path):
    """Draws red polygon(s) from label txt onto the image."""
    img = Image.open(img_path).convert('RGB')
    if not lbl_path.exists():
        print(f"[Warning] GT label {lbl_path.name} missing; showing raw image.")
        return img
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for line in open(lbl_path):
        vals = list(map(float, line.split()))
        seg  = vals[5:]
        poly = [(seg[i]*w, seg[i+1]*h) for i in range(0, len(seg), 2)]
        draw.polygon(poly, outline='red', width=3)
    return img

# choose the extension your val images use:
samples = sorted(IMG_VAL.glob('*.jpg'))  # or '*.png'

for img_path in samples:
    stem       = img_path.stem
    gt_lbl     = LBL_VAL       / f'{stem}.txt'
    pred_img_f = PRED_IMG_DIR  / f'{stem}.jpg'
    #pred_lbl_f = PRED_LBL_DIR  / f'{stem}.txt'  # only if you want to draw pred polys

    # 1) Load & overlay GT
    gt_img = draw_gt_seg(img_path, gt_lbl)

    # 2) Load predicted overlay JPG
    if not pred_img_f.exists():
        print(f"[Warning] pred image {pred_img_f.name} missing; skipping.")
        continue
    pred_img = Image.open(pred_img_f).convert('RGB')

    # 3) Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(gt_img)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    ax2.imshow(pred_img)
    ax2.set_title('Prediction')
    ax2.axis('off')

    plt.tight_layout()
    out_path = OUT_DIR / f'{stem}_gt_vs_pred.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

print(f"Done — saved {len(list(OUT_DIR.glob('*.png')))} plots to {OUT_DIR.resolve()}")
