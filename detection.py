from ultralytics import YOLO

def train_flake_detector(
    data_cfg: str = 'flake.yaml',
    pretrained_model: str = 'yolov8n.pt',
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    lr0: float = 0.01,
    project: str = 'flake_runs',
    run_name: str = 'exp1'
):

    model = YOLO(pretrained_model)
    model.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr0,
        project=project,
        name=run_name,
        cache=True,
        verbose=True
    )

if __name__ == '__main__':
    train_flake_detector()
