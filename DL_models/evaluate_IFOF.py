import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_IF import get_image_dataloaders       # For IF models
from data_IFOF_updated import get_dataloaders, get_class_weights  # For IFOF models
from model_compare import (ResNetModel_IF, EfficientNetModel_IF, DenseNetModel_IF, MobileNetModel_IF,
                          ResNetModel_IFOF, EfficientNetModel_IFOF, DenseNetModel_IFOF, MobileNetModel_IFOF)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set directories and batch size
image_dir = '/home/rdas/student_engagement/WACV'
feature_dir = '/home/rdas/student_engagement/WACV'
batch_size = 32  # Change as needed

# Load validation dataloaders
train_loader_IF, _, _ = get_image_dataloaders(image_dir, batch_size)
train_loader_IFOF, _, _ = get_dataloaders(image_dir, feature_dir, batch_size)

# Define evaluation functions for IF models
def evaluate_IF(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_labels, all_probs

# Define evaluation functions for IFOF models
def evaluate_IFOF(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, features, labels in data_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images.unsqueeze(1), features.unsqueeze(1))
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_labels, all_probs

# Helper functions to compute metrics for IF and IFOF models
def compute_metrics_IF(data_loader, model, criterion, device):
    loss, true_labels, probs = evaluate_IF(data_loader, model, criterion, device)
    preds = np.argmax(probs, axis=1)
    return {
        "loss": loss,
        "accuracy": accuracy_score(true_labels, preds),
        "precision": precision_score(true_labels, preds, average='weighted', zero_division=0),
        "recall": recall_score(true_labels, preds, average='weighted', zero_division=0),
        "f1": f1_score(true_labels, preds, average='weighted')
    }

def compute_metrics_IFOF(data_loader, model, criterion, device):
    loss, true_labels, probs = evaluate_IFOF(data_loader, model, criterion, device)
    preds = np.argmax(probs, axis=1)
    return {
        "loss": loss,
        "accuracy": accuracy_score(true_labels, preds),
        "precision": precision_score(true_labels, preds, average='weighted', zero_division=0),
        "recall": recall_score(true_labels, preds, average='weighted', zero_division=0),
        "f1": f1_score(true_labels, preds, average='weighted')
    }

# For simplicity, we use dummy class weights (ones) here.
num_classes = 3
dummy_weights = torch.ones(num_classes).float().to(device)
criterion_IF = nn.CrossEntropyLoss(weight=dummy_weights)
criterion_IFOF = nn.CrossEntropyLoss(weight=dummy_weights)

# Define paths for saved models
paths = {
    "ResNet_IF": "hyper-1/best_model_ResNet_IF.pth",
    "EfficientNet_IF": "hyper-1/best_model_EfficientNet_IF.pth",
    "DenseNet_IF": "hyper-1/best_model_DenseNet_IF.pth",
    "MobileNet_IF": "hyper-1/best_model_MobileNet_IF.pth",
    "ResNet_IFOF": "hyper-1/best_model_ResNet_IFOF.pth",
    "EfficientNet_IFOF": "hyper-1/best_model_EfficientNet_IFOF.pth",
    "DenseNet_IFOF": "hyper-1/best_model_DenseNet_IFOF.pth",
    "MobileNet_IFOF": "hyper-1/best_model_MobileNet_IFOF.pth"
}

# Load IF models
model_ResNet_IF = ResNetModel_IF(num_classes=num_classes, dropout_rate=0.5)
model_ResNet_IF.load_state_dict(torch.load(paths["ResNet_IF"], map_location=device))
model_ResNet_IF.to(device)

model_EfficientNet_IF = EfficientNetModel_IF(num_classes=num_classes, dropout_rate=0.5, freeze_features=True)
model_EfficientNet_IF.load_state_dict(torch.load(paths["EfficientNet_IF"], map_location=device))
model_EfficientNet_IF.to(device)

model_DenseNet_IF = DenseNetModel_IF(num_classes=num_classes, dropout_rate=0.5)
model_DenseNet_IF.load_state_dict(torch.load(paths["DenseNet_IF"], map_location=device))
model_DenseNet_IF.to(device)

model_MobileNet_IF = MobileNetModel_IF(num_classes=num_classes, dropout_rate=0.5)
model_MobileNet_IF.load_state_dict(torch.load(paths["MobileNet_IF"], map_location=device))
model_MobileNet_IF.to(device)

# Load IFOF models
model_ResNet_IFOF = ResNetModel_IFOF(dropout_rate=0.5)
model_ResNet_IFOF.load_state_dict(torch.load(paths["ResNet_IFOF"], map_location=device))
model_ResNet_IFOF.to(device)

model_EfficientNet_IFOF = EfficientNetModel_IFOF()
model_EfficientNet_IFOF.load_state_dict(torch.load(paths["EfficientNet_IFOF"], map_location=device))
model_EfficientNet_IFOF.to(device)

model_DenseNet_IFOF = DenseNetModel_IFOF(dropout_rate=0.5)
model_DenseNet_IFOF.load_state_dict(torch.load(paths["DenseNet_IFOF"], map_location=device))
model_DenseNet_IFOF.to(device)

model_MobileNet_IFOF = MobileNetModel_IFOF(dropout_rate=0.5)
model_MobileNet_IFOF.load_state_dict(torch.load(paths["MobileNet_IFOF"], map_location=device))
model_MobileNet_IFOF.to(device)

# Compute metrics for each model
metrics = {}
# IF models
metrics["ResNet_IF"] = compute_metrics_IF(train_loader_IF, model_ResNet_IF, criterion_IF, device)
metrics["EfficientNet_IF"] = compute_metrics_IF(train_loader_IF, model_EfficientNet_IF, criterion_IF, device)
metrics["DenseNet_IF"] = compute_metrics_IF(train_loader_IF, model_DenseNet_IF, criterion_IF, device)
metrics["MobileNet_IF"] = compute_metrics_IF(train_loader_IF, model_MobileNet_IF, criterion_IF, device)
# IFOF models
metrics["ResNet_IFOF"] = compute_metrics_IFOF(train_loader_IFOF, model_ResNet_IFOF, criterion_IFOF, device)
metrics["EfficientNet_IFOF"] = compute_metrics_IFOF(train_loader_IFOF, model_EfficientNet_IFOF, criterion_IFOF, device)
metrics["DenseNet_IFOF"] = compute_metrics_IFOF(train_loader_IFOF, model_DenseNet_IFOF, criterion_IFOF, device)
metrics["MobileNet_IFOF"] = compute_metrics_IFOF(train_loader_IFOF, model_MobileNet_IFOF, criterion_IFOF, device)

# Convert metrics dictionary to a pandas DataFrame
df_metrics = pd.DataFrame(metrics).T  # Transpose so each row is a model
df_metrics.index.name = "Model"
df_metrics.reset_index(inplace=True)
outputpath = "~/Student_Engagement/Results/DL"
# Save the metrics to CSV
csv_filename = os.path.join(outputpath,"evaluation_metrics_trainset.csv")
df_metrics.to_csv(csv_filename, index=False)
print(f"Saved evaluation metrics to {csv_filename}")
print(df_metrics)
