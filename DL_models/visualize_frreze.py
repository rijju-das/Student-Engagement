import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report
from data_IF import get_image_dataloaders, get_class_weights
from data_IFOF import get_dataloaders
from model_freeze import ResNetModel_IFOF, ResNetModel_IF, EfficientNetModel_IF, EfficientNetModel_IFOF
import pickle
import numpy as np

def evaluate_modelIFOF(model, loader):
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, features, labels in loader:
            outputs = model(images.unsqueeze(1), features.unsqueeze(1))
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # Print detailed classification report
    report = classification_report(all_labels, all_preds, target_names=['0', '1', '2'], zero_division=0)
    print("Classification Report:\n", report)
    
    return cm, accuracy, precision, recall, f1

def evaluate_modelIF(model, loader):
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # Print detailed classification report
    report = classification_report(all_labels, all_preds, target_names=['0', '1', '2'], zero_division=0)
    print("Classification Report:\n", report)
    
    return cm, accuracy, precision, recall, f1

def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('model/Fcm_IF_WACV.pdf')
    plt.show()


# Configuration
batch_size = 32

# Load Datasets
image_dir = '/home/rdas/student_engagement/WACV'
feature_dir = '/home/rdas/student_engagement/WACV'
_, _, test_loader_IF = get_image_dataloaders(image_dir, batch_size)
_, _, test_loader_IFOF = get_dataloaders(image_dir, feature_dir, batch_size)

# Load Model
modelResNet_IFOF = ResNetModel_IFOF()
modelResNet_IF = ResNetModel_IF()
modelEfficientNet_IFOF = EfficientNetModel_IFOF()
modelEfficientNet_IF = EfficientNetModel_IF()
modelResNet_IFOF.load_state_dict(torch.load('model/FResNetCNN_IFOF_WACV.pth'))
modelResNet_IF.load_state_dict(torch.load('model/FResNetCNN_IF_WACV.pth'))
modelEfficientNet_IFOF.load_state_dict(torch.load('model/FEfficientNet_IFOF_WACV.pth'))
modelEfficientNet_IF.load_state_dict(torch.load('model/FEfficientNet_IF_WACV.pth'))


# Load history
with open('model/FhistoryResNetCNN_IF_WACV.pkl', 'rb') as f:
    historyResNet_IF = pickle.load(f)
with open('model/FhistoryResNetCNN_IFOF_WACV.pkl', 'rb') as f:
    historyResNet_IFOF = pickle.load(f)
with open('model/FhistoryEfficientNet_IFOF_WACV.pkl', 'rb') as f:
    historyEffiNet_IFOF = pickle.load(f)
with open('model/FhistoryEfficientNet_IF_WACV.pkl', 'rb') as f:
    historyEffiNet_IF = pickle.load(f)

# Initialize lists
cm = []
accuracy = []
precision = []
recall = []
f1 = []

# Evaluate models and assign metrics
cm0, accuracy0, precision0, recall0, f10 = evaluate_modelIFOF(modelResNet_IFOF, test_loader_IFOF)
cm1, accuracy1, precision1, recall1, f11 = evaluate_modelIF(modelResNet_IF, test_loader_IF)
cm2, accuracy2, precision2, recall2, f12 = evaluate_modelIFOF(modelEfficientNet_IFOF, test_loader_IFOF)
cm3, accuracy3, precision3, recall3, f13 = evaluate_modelIF(modelEfficientNet_IF, test_loader_IF)

# Append results to lists
cm.append(cm0)
cm.append(cm1)
cm.append(cm2)
cm.append(cm3)

accuracy.append(accuracy0)
accuracy.append(accuracy1)
accuracy.append(accuracy2)
accuracy.append(accuracy3)

precision.append(precision0)
precision.append(precision1)
precision.append(precision2)
precision.append(precision3)

recall.append(recall0)
recall.append(recall1)
recall.append(recall2)
recall.append(recall3)

f1.append(f10)
f1.append(f11)
f1.append(f12)
f1.append(f13)

# Create a dictionary and then a DataFrame
eval_metrics = {
    'Model': ['ResNet_IFOF','ResNet_IF','EfficientNet_IFOF','EfficientNet_IF'],
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1': f1
}

import pandas as pd

df_metrics = pd.DataFrame(eval_metrics)
print(df_metrics)
df_metrics.to_csv("~Student_Engagement/Results/DL/Fevaluation.csv")



