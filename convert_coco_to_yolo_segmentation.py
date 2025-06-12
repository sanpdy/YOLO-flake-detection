import json
import random
from pathlib import Path

COCO_JSON   = Path('/home/sankalp/yolo_flake_detection/new_data/annotations/instances_default_clean.json')
IMG_ROOT    = Path('/home/sankalp/yolo_flake_detection/new_data/')  
YOLO_ROOT   = Path('/home/sankalp/yolo_flake_detection/new_data/yolo_dataset_segmentation')
TRAIN_RATIO = 0.8

data = json.loads(COCO_JSON.read_text())
images = {img['id']: img for img in data['images']}

for split in ('train','val'):
    (YOLO_ROOT/'images'/split).mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT/'labels'/split).mkdir(parents=True, exist_ok=True)

all_ids = list(images.keys())
random.shuffle(all_ids)
cut = int(len(all_ids) * TRAIN_RATIO)
split_of = {img_id: ('train' if i < cut else 'val')
            for i, img_id in enumerate(all_ids)}

yolo_labels = {img_id: [] for img_id in all_ids}

for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in images:
        continue

    img = images[img_id]
    w, h = img['width'], img['height']

    seg = ann.get('segmentation')
    if isinstance(seg, dict):
        continue

    if not isinstance(seg, list) or len(seg) == 0:
        continue

    coords = seg[0] if isinstance(seg[0], list) else seg

    if len(coords) < 6:
        print(f"Skipping image {img_id} with insufficient segmentation points.")
        continue

    yolo_points = []
    for i in range(0, len(coords), 2):
        x_norm = coords[i]   / w
        y_norm = coords[i+1] / h
        yolo_points.append(f"{x_norm:.6f}")
        yolo_points.append(f"{y_norm:.6f}")

    if yolo_points:
        yolo_line = "0 " + " ".join(yolo_points)
        yolo_labels[img_id].append(yolo_line)

for img_id, img in images.items():
    split = split_of[img_id]
    src_img = Path(img['file_name'])
    dst_img = YOLO_ROOT/'images'/split/src_img.name
    dst_lbl = YOLO_ROOT/'labels'/split/(src_img.stem + '.txt')
    
    if src_img.exists():
        dst_img.write_bytes(src_img.read_bytes())
        
        if yolo_labels[img_id]:
            dst_lbl.write_text('\n'.join(yolo_labels[img_id]))
    else:
        print(f"Warning: Image {src_img} not found!")

print("\nConversion complete!")
print(f"- Images in:  {YOLO_ROOT/'images'}")
print(f"- Labels in:  {YOLO_ROOT/'labels'}")