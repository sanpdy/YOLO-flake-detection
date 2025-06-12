from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path

IMG_VAL   = Path('/home/sankalp/yolo_flake_detection/new_data/yolo_dataset_segmentation/images/train')
LBL_VAL   = Path('/home/sankalp/yolo_flake_detection/new_data/yolo_dataset_segmentation/labels/val')
PRED_DIR  = Path('/home/sankalp/yolo_flake_detection/inference_segmentation_results/seg_outputs')
OUT_DIR   = Path('gt_pred_segmentation')
OUT_DIR.mkdir(exist_ok=True)

def draw_gt_seg(img_path, lbl_path):
    img = Image.open(img_path).convert('RGB')
    if not lbl_path.exists():
        print(f"[Warning] label file {lbl_path} not found; returning raw image.")
        return img

    w, h = img.size
    draw = ImageDraw.Draw(img)
    with open(lbl_path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            seg = vals[5:]
            poly = [(seg[i] * w, seg[i+1] * h) for i in range(0, len(seg), 2)]
            draw.polygon(poly, outline='red', width=3)
    return img


samples = sorted(IMG_VAL.glob('*.jpg'))

for img_path in samples:
    stem     = img_path.stem
    lbl_file = LBL_VAL  / f'{stem}.txt'
    pred_file= PRED_DIR / f'{stem}.jpg'

    # 1) Ground-truth overlay (handles missing .txt inside draw_gt_seg already)
    gt_img = draw_gt_seg(img_path, lbl_file)

    # 2) Check for the prediction image
    if not pred_file.exists():
        print(f"[Warning] prediction file {pred_file} not found; skipping this sample.")
        continue  # or load a blank image: Image.open(img_path) …
    pred_img = Image.open(pred_file).convert('RGB')

    # 3) Plot side by side
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1.imshow(gt_img);   ax1.set_title('Ground Truth Seg'); ax1.axis('off')
    ax2.imshow(pred_img); ax2.set_title('Prediction');    ax2.axis('off')
    plt.tight_layout()

    out_path = OUT_DIR / f'{stem}_gt_vs_pred.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

print(f"Done – saved plots to {OUT_DIR.resolve()}")