import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ResNetModel_IF(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetModel_IF, self).__init__()
        self.image_cnn = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.image_cnn.fc = nn.Identity()  # Remove the last fully connected layer
        
    def forward(self, images):
        x = self.image_cnn(images)
        return x

class EfficientNetModel_IF(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientNetModel_IF, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()  # Remove the last fully connected layer

    def forward(self, images):
        x = self.efficientnet(images)
        return x


class OpenFaceModel(nn.Module):
    def __init__(self, feature_input_dim=709, feature_output_dim=128):
        super(OpenFaceModel, self).__init__()
        self.fc1 = nn.Linear(feature_input_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, feature_output_dim)  # Adjusted output dim to 128
        self.dropout2 = nn.Dropout(0.5)
    
    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return x

class ResNetModel_shared(nn.Module):
    def __init__(self, num_classes=3, image_output_dim=512, feature_output_dim=128):
        super(ResNetModel_shared, self).__init__()
        self.image_model = ResNetModel_IF()
        self.openface_model = OpenFaceModel(feature_output_dim=feature_output_dim)
        
        combined_input_dim = image_output_dim + feature_output_dim  # ResNet18 output + OpenFaceModel output
        self.shared_fc1 = nn.Linear(combined_input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.shared_fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, num_classes)
    
    def forward(self, images, features):
        image_features = self.image_model(images)
        openface_features = self.openface_model(features)
        
        combined_features = torch.cat((image_features, openface_features), dim=1)
        
        x = F.relu(self.shared_fc1(combined_features))
        x = self.dropout1(x)
        x = F.relu(self.shared_fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        return x

class EfficientnetModel_shared(nn.Module):
    def __init__(self, num_classes=3, image_output_dim=1280, feature_output_dim=128):
        super(EfficientnetModel_shared, self).__init__()
        self.image_model = EfficientNetModel_IF()
        self.openface_model = OpenFaceModel(feature_output_dim=feature_output_dim)
        
        combined_input_dim = image_output_dim + feature_output_dim  # EfficienNet output + OpenFaceModel output
        self.shared_fc1 = nn.Linear(combined_input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.shared_fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, num_classes)
    
    def forward(self, images, features):
        image_features = self.image_model(images)
        openface_features = self.openface_model(features)
        
        combined_features = torch.cat((image_features, openface_features), dim=1)
        
        x = F.relu(self.shared_fc1(combined_features))
        x = self.dropout1(x)
        x = F.relu(self.shared_fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        return x
