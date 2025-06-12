from ultralytics import YOLO

model = YOLO('yolo11x-seg.pt')
results = model.train(data="flake_segmentation.yaml", epochs=100, imgsz=640, batch=16, lr0=0.01, project='flake_segmentation_runs', name='exp1', cache=True, verbose=True)
