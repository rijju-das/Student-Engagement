import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ResNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=512, feature_input_dim=709, num_classes=3):
        super(ResNetModel_IFOF, self).__init__()
        self.image_cnn = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.image_cnn.fc = nn.Identity()  # Remove the last fully connected layer

        # Calculate the expected input dimension for the first fully connected layer
        combined_input_dim = image_input_dim + feature_input_dim  # 512 + 709 = 1221

        self.fc1 = nn.Linear(combined_input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes) 
        
    def forward(self, images, features):
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.image_cnn(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Get the last output from the CNN

        features = features.view(batch_size, -1)  # Flatten the features
        combined_features = torch.cat((image_features, features), dim=1)
        
        x = F.relu(self.fc1(combined_features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class ResNetModel_IF(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetModel_IF, self).__init__()
        self.image_cnn = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.image_cnn.fc = nn.Linear(self.image_cnn.fc.in_features, num_classes)  
        
        # Dropout layers for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        # images shape should be (batch_size, channels, height, width)
        x = self.image_cnn(images)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class EfficientNetModel_IF(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientNetModel_IF, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        # Dropout layers for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        x = self.efficientnet(images)
        x = self.dropout(x)
        return x

class EfficientNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=1280, feature_input_dim=709, num_classes=3):
        super(EfficientNetModel_IFOF, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()  # Remove the last fully connected layer

        # Calculate the expected input dimension for the first fully connected layer
        combined_input_dim = image_input_dim + feature_input_dim  # 1280 + 709 = 1989

        self.fc1 = nn.Linear(combined_input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes) 
        
    def forward(self, images, features):
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.efficientnet(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Get the last output from the CNN

        features = features.view(batch_size, -1)  # Flatten the features
        combined_features = torch.cat((image_features, features), dim=1)
        
        x = F.relu(self.fc1(combined_features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
