from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path

IMG_DIR = Path('/home/sankalp/yolo_flake_detection/new_data/yolo_dataset_segmentation/images/train')
LBL_DIR = Path('/home/sankalp/yolo_flake_detection/new_data/yolo_dataset_segmentation/labels/train')
OUT_DIR = Path('gt_only_segmentation')
OUT_DIR.mkdir(exist_ok=True)

# gather all label stems
label_paths = sorted(LBL_DIR.glob('*.txt'))
label_stems = {p.stem for p in label_paths}

# gather all image files (jpg + png)
img_paths = sorted(
    [*IMG_DIR.glob('*.png'), *IMG_DIR.glob('*.jpg')]
)
img_stems = {p.stem for p in img_paths}

# find intersection
common = sorted(label_stems & img_stems)
print(f"Found {len(img_paths)} images, {len(label_paths)} label files, {len(common)} matching pairs")

def draw_gt(img_p, lbl_p):
    img = Image.open(img_p).convert('RGB')
    w,h = img.size
    draw = ImageDraw.Draw(img)
    for ln in open(lbl_p):
        vals = list(map(float, ln.split()))
        seg = vals[5:]
        poly = [(seg[i]*w, seg[i+1]*h) for i in range(0,len(seg),2)]
        draw.polygon(poly, outline='red', width=3)
    return img

for stem in common:
    # find your image path
    # (we know one of these exists because of the intersection)
    for ext in ('png','jpg'):
        img_file = IMG_DIR / f"{stem}.{ext}"
        if img_file.exists(): break
    lbl_file = LBL_DIR / f"{stem}.txt"

    gt = draw_gt(img_file, lbl_file)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(gt); ax.axis('off'); ax.set_title(f"{stem}  GT")
    out_path = OUT_DIR / f"{stem}_gt.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

print("Done — saved", len(common), "ground-truth overlays in", OUT_DIR)
