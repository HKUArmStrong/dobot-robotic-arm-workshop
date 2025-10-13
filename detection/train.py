# YOLOv8 Ultralytics training script
# Trains a YOLO model using a dataset in YOLO format under dataset/generated
# Generates utils as needed for dataset config and training

import os
from ultralytics import YOLO
from roboflow import Roboflow

# Download dataset from Roboflow if not already present
def download_dataset():
    HOME = os.path.expanduser("~")
    datasets_dir = os.path.join(HOME, "datasets_yolo")
    os.makedirs(datasets_dir, exist_ok=True)
    os.chdir(datasets_dir)

    # Download the dataset from Roboflow (if available)
    datasetname = "dog_cat-test"  # Change to your dataset name
    # Default path where Roboflow will put the downloaded YOLO dataset
    local_dataset_dir = os.path.join(datasets_dir, datasetname + "-1")
    if os.path.isdir(local_dataset_dir):
        print(f"Dataset already exists at {local_dataset_dir}")
        return local_dataset_dir

    # Replace the API key, workspace and version as appropriate for your project
    rf = Roboflow(api_key="f5pOg5frjQ3XAImBbLxX")
    project = rf.workspace("ravensburgweingarten").project(datasetname)
    dataset = project.version(1).download("yolov11")
    # Roboflow returns an object whose location attribute points to dataset path
    if hasattr(dataset, 'location'):
        return dataset.location
    return local_dataset_dir

# Main training function
def main():
    dataset_path = download_dataset()

    # Setup some sensible defaults and paths
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_dir = os.path.join(repo_dir, "yolo")
    os.makedirs(yolo_dir, exist_ok=True)

    # data yaml path expected by Ultralytics YOLO - require it to exist
    data_yaml_path = os.path.join(dataset_path, "data.yaml")

    # Train YOLOv11 (medium) model, change model as needed
    model = YOLO('yolo11m.pt')
    results = model.train(
        data=data_yaml_path,
        epochs=1000,
        imgsz=640,
        batch=4,
        workers=16,
        project=os.path.join(repo_dir, 'model'),
        name='yolo11m-screw',
        save_period=50,  # Save every 50 epochs
    )

if __name__ == "__main__":
    main()