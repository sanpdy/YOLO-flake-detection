from ultralytics.data.converter import convert_coco

# For keypoints data (like person_keypoints_val2017.json)
convert_coco(
    labels_dir="/home/sankalp/yolo_flake_detection/monark_data/annotations",  # Directory containing your json file
    save_dir="/home/sankalp/yolo_flake_detection/monark_data/yolo/annotations",
    use_segments =True
)