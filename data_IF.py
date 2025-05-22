import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np

class WACVImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in ['0', '1', '2']:
            label_dir = os.path.join(image_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    if os.path.exists(image_path):
                        self.image_paths.append(image_path)
                        self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_class_weights(labels):
    # Calculate the number of samples for each class
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    
    # Calculate weights as the inverse of the number of samples for each class
    weight = 1. / class_sample_count
    
    # Normalize the weights to make sure they sum to 1
    normalized_weight = weight / weight.sum()
    # Convert the weights to a tensor
    class_weights = torch.tensor(normalized_weight, dtype=torch.float32)
    
    return class_weights

def get_image_dataloaders(image_dir, batch_size=32, val_split=0.2, test_split=0.1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.44730392, 0.43107784, 0.42404088], std=[0.1695384, 0.16244154, 0.16137291]),
    ])
    
    dataset = WACVImageDataset(image_dir, transform=transform)
    
    val_size = int(len(dataset) * val_split)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_labels = np.array([dataset.labels[i] for i in train_dataset.indices])
    train_weights = get_class_weights(train_labels)
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

