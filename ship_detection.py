#!/usr/bin/env python3

# Import required libraries
import torch
import numpy as np
import pandas as pd
import os
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

def train_model(model, device):
    """Train the YOLO model with GPU optimization"""
    # Set GPU batch size based on available memory
    gpu_batch_size = 16  # You can adjust this based on your GPU memory
    
    # Start the training session with GPU optimization
    results = model.train(
        data='/home/mozer/projects/yolo_ship/data.yaml',   # Path to your ship dataset configuration
        epochs=100,                       # Number of epochs to train
        imgsz=640,                        # Image size
        batch=gpu_batch_size if torch.cuda.is_available() else 8,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=4 if torch.cuda.is_available() else 4,
        amp=True,                         # Automatic Mixed Precision for faster training
        **get_aggressive_hyperparameters(),
        name='ship_detection_aggressive_aug'
    )
    
    return results

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
    # Setup CUDA and get device
    device = setup_cuda()
    
    # Set visualization style
    sns.set_style('darkgrid')
    
    # Load the model
    model = YOLO("/home/mozer/projects/yolo_ship/yolo11n.pt")
    model = model.to(device)
    
    # Train the model
    results = train_model(model, device)
    
    # Validate the model
    metrics = validate_model(model)
    
    # Save the model
    save_model(model)

if __name__ == "__main__":
    main()