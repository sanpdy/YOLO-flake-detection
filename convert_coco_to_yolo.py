#!/usr/bin/env python3
import json
import random
from pathlib import Path

COCO_JSON   = Path('/home/sankalp/yolo_flake_detection/data/annotations/instances_default_clean.json')
IMG_ROOT    = Path('/home/sankalp/yolo_flake_detection/data')  
YOLO_ROOT   = Path('/home/sankalp/yolo_flake_detection/data/yolo_dataset')
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
    img = images[ann['image_id']]
    w, h = img['width'], img['height']
    x_min, y_min, bw, bh = ann['bbox']
    x_c = (x_min + bw/2) / w
    y_c = (y_min + bh/2) / h
    bw_n = bw / w
    bh_n = bh / h
    yolo_labels[ann['image_id']].append(
        f"0 {x_c:.6f} {y_c:.6f} {bw_n:.6f} {bh_n:.6f}"
    )

for img_id, img in images.items():
    split = split_of[img_id]
    src_img = Path(img['file_name'])
    dst_img = YOLO_ROOT/'images'/split/src_img.name
    dst_lbl = YOLO_ROOT/'labels'/split/(src_img.stem + '.txt')
    dst_img.write_bytes(src_img.read_bytes())
    dst_lbl.write_text('\n'.join(yolo_labels[img_id]))

print("\n Conversion complete!")
print(f"- Images in:  {YOLO_ROOT/'images'}")
print(f"- Labels in:  {YOLO_ROOT/'labels'}")
