#!/usr/bin/env python3

# Import required libraries
import torch
import numpy as np
import pandas as pd
import os
import argparse
import tempfile
import shutil
import yaml
from ultralytics import YOLO
import squarify
import matplotlib.pyplot as plt
import cv2
import random
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.cluster import KMeans
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

def setup_cuda():
    """Configure CUDA settings for PyTorch"""
    torch.cuda.empty_cache()  # Clear GPU memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        # Print GPU info
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Set PyTorch to use the GPU
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
    
    return device

def load_test_data():
    """Load and process test images and labels"""
    path = Path("/home/mozer/projects/yolo_ship/archive/ships-aerial-images/test")
    test_images = []
    test_labels = []
    
    for subfolder in path.iterdir():
        if subfolder.is_dir():
            for file in subfolder.iterdir():
                if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:  # Image files
                    test_images.append(mpimg.imread(file))
                elif file.suffix.lower() == '.txt':  # Label files
                    test_labels.append(file)
    
    return test_images, test_labels

def visualize_training_samples():
    """Visualize random training samples with bounding boxes"""
    # Define the paths to the images and labels directories
    train_images = "/home/mozer/projects/yolo_ship/archive/ships-aerial-images/train/images"
    train_labels = "/home/mozer/projects/yolo_ship/archive/ships-aerial-images/train/labels"
    
    # Get a list of all the image files in the training images directory
    image_files = os.listdir(train_images)
    
    # Choose 16 random image files from the list
    random_images = random.sample(image_files, 16)
    
    # Set up the plot
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    
    # Loop over the random images and plot the object detections
    for i, image_file in enumerate(random_images):
        row = i // 4
        col = i % 4
        
        # Load the image
        image_path = os.path.join(train_images, image_file)
        image = cv2.imread(image_path)
        
        # Load the labels for this image
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(train_labels, label_file)
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")
        
        # Loop over the labels and plot the object detections
        for label in labels:
            if len(label.split()) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, label.split())
            x_min = int((x_center - width/2) * image.shape[1])
            y_min = int((y_center - height/2) * image.shape[0])
            x_max = int((x_center + width/2) * image.shape[1])
            y_max = int((y_center + height/2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        # Show the image with the object detections
        axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[row, col].axis('off')
    
    plt.show()

def get_aggressive_hyperparameters():
    """Define aggressive hyperparameters for data augmentation"""
    return {
        # Geometric Augmentation
        'scale': 0.2,       # Aggressive scaling down (allows ships to be 20% of original size)
        'degrees': 15.0,    # Increased rotation (up to +/- 15 degrees)
        'shear': 5.0,       # Increased shear (up to +/- 5 degrees)
        'perspective': 0.0005, # Small perspective change
        
        # Photometric Augmentation (HSV)
        'hsv_h': 0.1,       # Hue adjustment
        'hsv_s': 0.7,       # Aggressive Saturation adjustment (simulates haze/glare)
        'hsv_v': 0.5,       # Aggressive Brightness (Value) adjustment (simulates varied lighting)
        
        # Advanced Augmentation
        'mosaic': 0.8,      # High probability for Mosaic (combines 4 images)
        'mixup': 0.1,       # Probability for MixUp (blends 2 images)
    }

def train_model(model, device, data_yaml='data.yaml', epochs=100):
    """Train the YOLO model with GPU optimization and augmentation.

    Determines a safe batch size from available GPU memory when a GPU is present
    and passes the augmentation hyperparameters (hyp) returned by
    `get_aggressive_hyperparameters()` into the trainer.
    """

    # Determine device/AMP and batch size
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1e9
        # Heuristic batch sizing based on GPU memory
        if mem_gb >= 16:
            batch = 16
        elif mem_gb >= 8:
            batch = 8
        else:
            # For small GPUs (e.g., 6GB) use a conservative batch
            batch = 2
        device_arg = 0
        amp_flag = True
    else:
        batch = 4
        device_arg = 'cpu'
        amp_flag = False

    # Workers: keep reasonable default, user can override in the script
    workers = min(8, max(2, (os.cpu_count() or 4) // 2))

    # If GPU memory is small, reduce image size and workers, avoid RAM caching
    if torch.cuda.is_available() and mem_gb < 8:
        imgsz = 416
        # further reduce workers to avoid memory pressure
        workers = min(workers, 4)
        cache_mode = 'disk'  # avoid using too much RAM for cache
    else:
        imgsz = 640
        cache_mode = True

    # Augmentation hyperparameters
    aug = get_aggressive_hyperparameters()

    # Ultralytics train() does not accept a single 'hyp' dict here; instead pass
    # augmentation keys individually. Filter augmentation keys to a whitelist
    # accepted by the trainer to avoid argument errors.
    valid_aug_keys = {
        'scale', 'degrees', 'shear', 'perspective',
        'hsv_h', 'hsv_s', 'hsv_v', 'mosaic', 'mixup'
    }
    aug_kwargs = {k: v for k, v in aug.items() if k in valid_aug_keys}

    # set allocation conf to reduce fragmentation when necessary
    if torch.cuda.is_available():
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    results = model.train(
        data=data_yaml,             # Dataset configuration file
        epochs=epochs,              # Number of epochs to train
        imgsz=imgsz,                # Image size (may be reduced on small GPUs)
        batch=batch,                # Computed batch size
        device=device_arg,          # GPU (int) or 'cpu'
        workers=workers,            # Number of worker threads
        amp=amp_flag,               # Automatic Mixed Precision when GPU available
        single_cls=True,            # Single class detection (ships only)
        rect=True,                  # Rectangular training
        cos_lr=True,                # Cosine LR scheduler
        close_mosaic=10,            # Disable mosaic augmentation for final epochs
        cache=cache_mode,           # Cache mode (disk or True)
        save=True,                  # Save checkpoints
        **aug_kwargs,               # Unpack allowed augmentation args
        name='ship_detection_aggressive_aug',
        exist_ok=True
    )

    return results


def create_subset_dataset(src_data_yaml: str, subset_size: int, tmpdir: str | None = None):
    """Create a temporary dataset using only a subset of training images.

    Returns (temp_data_yaml_path, tmpdir_path). Caller should remove tmpdir when done.
    """
    if subset_size <= 0:
        return src_data_yaml, None

    # Load the source data.yaml
    with open(src_data_yaml, 'r') as f:
        cfg = yaml.safe_load(f)

    train_path = cfg.get('train')
    val_path = cfg.get('val')
    test_path = cfg.get('test')
    nc = cfg.get('nc', 1)
    names = cfg.get('names', ['ship'])

    train_images_dir = Path(train_path)
    # guess labels dir by replacing 'images' with 'labels' if present
    if 'images' in str(train_images_dir):
        train_labels_dir = Path(str(train_images_dir).replace('/images', '/labels'))
    else:
        train_labels_dir = train_images_dir.parent / 'labels'

    # collect image files
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_images = [p for p in sorted(train_images_dir.iterdir()) if p.suffix.lower() in exts]
    if len(all_images) == 0:
        raise RuntimeError(f'No training images found in {train_images_dir}')

    subset_size = min(subset_size, len(all_images))
    random.seed(42)
    subset = random.sample(all_images, subset_size)

    # create temp dir
    tmpdir_path = Path(tmpdir) if tmpdir else Path(tempfile.mkdtemp(prefix='yolo_subset_'))
    tmp_train_images = tmpdir_path / 'train' / 'images'
    tmp_train_labels = tmpdir_path / 'train' / 'labels'
    tmp_train_images.mkdir(parents=True, exist_ok=True)
    tmp_train_labels.mkdir(parents=True, exist_ok=True)

    # copy subset images and labels
    for img_path in subset:
        dst_img = tmp_train_images / img_path.name
        shutil.copy2(img_path, dst_img)
        label_name = img_path.with_suffix('.txt').name
        src_label = train_labels_dir / label_name
        if src_label.exists():
            shutil.copy2(src_label, tmp_train_labels / label_name)
        else:
            # create empty label file to avoid loader errors
            (tmp_train_labels / label_name).write_text('')

    # write new data.yaml
    tmp_data = {
        'train': str(tmp_train_images),
        'val': val_path,
        'test': test_path,
        'nc': nc,
        'names': names
    }

    tmp_data_yaml = tmpdir_path / 'data.yaml'
    with open(tmp_data_yaml, 'w') as f:
        yaml.safe_dump(tmp_data, f)

    return str(tmp_data_yaml), str(tmpdir_path)

def validate_model(model):
    """Validate the trained model"""
    # Validation with GPU
    print("\nRunning validation...")
    metrics = model.val(conf=0.25, split='test', device=0 if torch.cuda.is_available() else 'cpu')
    
    # Get validation metrics
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print("\nValidation Metrics:")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")
    print(f"Maps: {metrics.box.maps}")
    
    return metrics

def save_model(model):
    """Save the trained model in both PyTorch and ONNX formats"""
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save PyTorch model
    torch_path = os.path.join(save_dir, "ship_detector_final.pt")
    model.save(torch_path)
    print(f"Model saved to {torch_path}")
    
    # Export to ONNX format (useful for deployment)
    onnx_path = os.path.join(save_dir, "ship_detector_final.onnx")
    model.export(format="onnx", imgsz=640)
    print(f"Model exported to ONNX format at {onnx_path}")

def main():
    """Main execution function"""
    # Setup device and print info
    device = setup_cuda()

    # Set visualization style
    sns.set_style('darkgrid')
    parser = argparse.ArgumentParser(description='Train YOLO ship detector')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--subset', type=int, default=0, help='If >0, train on a subset of this many training images')
    parser.add_argument('--no-clean', action='store_true', help="Don't delete temporary subset dir")
    args = parser.parse_args()

    try:
        # Try to load the requested weights. If local 'yolo11x.pt' is not present
        # attempt to load 'yolov8x.pt' which ultralytics can download automatically.
        try:
            print("Attempting to load 'yolo11x.pt' (local)")
            model = YOLO('yolo11x.pt')
            print("Loaded local 'yolo11x.pt'")
        except Exception as e_local:
            print("Local 'yolo11x.pt' not available or failed to load:", e_local)
            try:
                print("Falling back to 'yolov8x.pt' (will download if needed)")
                model = YOLO('yolov8x.pt')
                print("Loaded 'yolov8x.pt' (remote)")
            except Exception as e_remote:
                # As last resort try smaller model to reduce memory usage
                try:
                    print("Falling back to 'yolov8n.pt' (nano)")
                    model = YOLO('yolov8n.pt')
                    print("Loaded 'yolov8n.pt' (remote)")
                except Exception:
                    raise RuntimeError(
                        "Failed to load any YOLO weights: tried 'yolo11x.pt', 'yolov8x.pt', and 'yolov8n.pt'. "
                        "If you're offline, place your model file (yolo11x.pt) in the project root or specify a path."
                    )

        # Possibly create a temporary subset dataset
        data_yaml_path = 'data.yaml'
        tmpdir = None
        if args.subset and args.subset > 0:
            print(f'Creating dataset subset of {args.subset} images...')
            data_yaml_path, tmpdir = create_subset_dataset('data.yaml', args.subset)

        # Train using train_model which handles GPU/augmentation logic
        print('Starting training (GPU if available)...')
        results = train_model(model, device, data_yaml=data_yaml_path, epochs=args.epochs)

        print('Training completed successfully!')

        # Validate and save
        validate_model(model)
        save_model(model)
        # cleanup
        if tmpdir and not args.no_clean:
            try:
                shutil.rmtree(tmpdir)
                print(f'Removed temporary dataset dir {tmpdir}')
            except Exception:
                print(f'Could not remove temporary dir {tmpdir}; you can remove it manually')
    except Exception as e:
        print(f'An error occurred during training: {str(e)}')
        raise

if __name__ == "__main__":
    main()