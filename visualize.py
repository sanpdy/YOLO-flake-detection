from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path

IMG_VAL   = Path('/home/sankalp/yolo_flake_detection/data/yolo_dataset/images/val')
LBL_VAL   = Path('/home/sankalp/yolo_flake_detection/data/yolo_dataset/labels/val')
PRED_DIR  = Path('inference_results/exp12_val_results')
OUT_DIR   = Path('gt_pred')
OUT_DIR.mkdir(exist_ok=True)

def draw_gt(img_path, lbl_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for line in open(lbl_path):
        cls, x_c, y_c, w_n, h_n = map(float, line.split())
        x_min = (x_c - w_n/2)*w
        y_min = (y_c - h_n/2)*h
        x_max = (x_c + w_n/2)*w
        y_max = (y_c + h_n/2)*h
        draw.rectangle([x_min,y_min,x_max,y_max], outline='red', width=3)
    return img

samples = (IMG_VAL.glob('*.jpg'))

for img_path in samples:
    stem = img_path.stem
    gt_img   = draw_gt(img_path, LBL_VAL/ (stem + '.txt'))
    pred_img = Image.open(PRED_DIR/ (stem + '.jpg')).convert('RGB')
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1.imshow(gt_img);   ax1.set_title('Ground Truth'); ax1.axis('off')
    ax2.imshow(pred_img); ax2.set_title('Prediction');   ax2.axis('off')
    plt.tight_layout()
    out_path = OUT_DIR / f'{stem}_gt_vs_pred.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

print(f"Saved plots to {OUT_DIR.resolve()}")
