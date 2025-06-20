import os
import json

base_data_path = "/home/sankalp/yolo_flake_detection/"
datasets = ['new_data']

all_images = []
all_annotations = []
current_image_id = 1
current_ann_id = 1

categories = None

for dataset in datasets:
    dataset_path = os.path.join(base_data_path, dataset)
    annotations_file = os.path.join(dataset_path, "annotations", "instances_default.json")
    images_dir = os.path.join(dataset_path, "images")
    
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found for {dataset} at {annotations_file}. Skipping.")
        continue

    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    if categories is None and "categories" in data:
        data["categories"] = [{"id": 0, "name": "flake"}]
        categories = data["categories"]
        print(data["categories"])

    image_id_map = {}
    for img in data.get('images', []):
        full_image_path = os.path.join(images_dir, img['file_name'])
        if os.path.exists(full_image_path):
            img['file_name'] = full_image_path
            image_id_map[img['id']] = current_image_id
            img['id'] = current_image_id
            current_image_id += 1
            all_images.append(img)
        else:
            print(f"Image {full_image_path} not found. Skipping.")

    for ann in data.get('annotations', []):
        old_image_id = ann['image_id']
        if old_image_id in image_id_map:
            ann['image_id'] = image_id_map[old_image_id]
            ann['id'] = current_ann_id  
            current_ann_id += 1
            all_annotations.append(ann)

merged_data = {
    "images": all_images,
    "annotations": all_annotations,
    "categories": categories if categories is not None else []
}

output_dir = os.path.join(base_data_path, "annotations")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "instances_default_clean.json")

with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)

print(f"Merged annotation file saved at {output_file}")
