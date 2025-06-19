import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, densenet121, mobilenet_v2
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

# Fusion module remains unchanged
class FusionAttention(nn.Module):
    def __init__(self, image_dim, feature_dim, fusion_dim):
        """
        Projects image and extra features to a common space (fusion_dim) and
        computes attention weights to fuse them.
        """
        super(FusionAttention, self).__init__()
        self.proj_img = nn.Linear(image_dim, fusion_dim)
        self.proj_feat = nn.Linear(feature_dim, fusion_dim)
        self.attn_fc = nn.Linear(fusion_dim * 2, 2)  # Outputs a weight for each modality

    def forward(self, image_features, extra_features):
        # image_features: (batch_size, image_dim)
        # extra_features: (batch_size, feature_dim)
        proj_img = torch.tanh(self.proj_img(image_features))    # (batch_size, fusion_dim)
        proj_feat = torch.tanh(self.proj_feat(extra_features))    # (batch_size, fusion_dim)
        combined = torch.cat([proj_img, proj_feat], dim=1)         # (batch_size, fusion_dim*2)
        weights = torch.softmax(self.attn_fc(combined), dim=1)     # (batch_size, 2)
        # Weighted sum of the projected features
        fused = weights[:, 0:1] * proj_img + weights[:, 1:2] * proj_feat
        return fused

######################################
# ResNet based models (existing)
######################################
class ResNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=512, feature_input_dim=709, num_classes=3, fusion_dim=256, dropout_rate=0.5):
        """
        Uses a ResNet18 backbone for image features and fuses with extra features using attention.
        """
        super(ResNetModel_IFOF, self).__init__()
        self.image_cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_cnn.fc = nn.Identity()  # Remove the final FC layer
        
        # Use FusionAttention instead of simple concatenation
        self.fusion = FusionAttention(image_dim=image_input_dim,
                                      feature_dim=feature_input_dim,
                                      fusion_dim=fusion_dim)
        
        # Build the classifier on the fused features
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, images, features):
        # images shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.image_cnn(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Using the last output
        
        features = features.view(batch_size, -1)  # Flatten extra features
        
        # Fuse the modalities using the attention module
        fused_features = self.fusion(image_features, features)
        
        x = F.relu(self.fc1(fused_features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class ResNetModel_IF(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetModel_IF, self).__init__()
        self.image_cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace final FC layer with one that produces num_classes outputs
        self.image_cnn.fc = nn.Linear(self.image_cnn.fc.in_features, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        x = self.image_cnn(images)
        x = F.relu(x)
        x = self.dropout(x)
        return x

######################################
# EfficientNet based models (existing)
######################################
class EfficientNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=1280, feature_input_dim=709, num_classes=3, fusion_dim=256):
        """
        Uses an EfficientNet-B0 backbone for image features and fuses with extra features using attention.
        """
        super(EfficientNetModel_IFOF, self).__init__()
        from torchvision.models import efficientnet_b0
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()  # Remove the final classifier
        
        self.fusion = FusionAttention(image_dim=image_input_dim,
                                      feature_dim=feature_input_dim,
                                      fusion_dim=fusion_dim)
        
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, images, features):
        # images shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.efficientnet(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Get the last output

        features = features.view(batch_size, -1)  # Flatten extra features
        
        fused_features = self.fusion(image_features, features)
        
        x = F.relu(self.fc1(fused_features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class EfficientNetModel_IF(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientNetModel_IF, self).__init__()
        from torchvision.models import efficientnet_b0
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        x = self.efficientnet(images)
        x = self.dropout(x)
        return x

######################################
# DenseNet based models (new)
######################################
class DenseNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=1024, feature_input_dim=709, num_classes=3, fusion_dim=256, dropout_rate=0.5):
        """
        Uses a DenseNet121 backbone for image features and fuses with extra features using attention.
        Note: DenseNet121's classifier typically outputs 1024 features.
        """
        super(DenseNetModel_IFOF, self).__init__()
        from torchvision.models import densenet121
        self.image_cnn = densenet121(pretrained=True)
        self.image_cnn.classifier = nn.Identity()  # Remove the classifier
        
        self.fusion = FusionAttention(image_dim=image_input_dim,
                                      feature_dim=feature_input_dim,
                                      fusion_dim=fusion_dim)
        
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, images, features):
        # images shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.image_cnn(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Use last output
        
        features = features.view(batch_size, -1)  # Flatten extra features
        
        fused_features = self.fusion(image_features, features)
        
        x = F.relu(self.fc1(fused_features))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class DenseNetModel_IF(nn.Module):
    def __init__(self, num_classes=3):
        super(DenseNetModel_IF, self).__init__()
        from torchvision.models import densenet121
        self.image_cnn = densenet121(pretrained=True)
        num_ftrs = self.image_cnn.classifier.in_features
        self.image_cnn.classifier = nn.Linear(num_ftrs, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        x = self.image_cnn(images)
        x = F.relu(x)
        x = self.dropout(x)
        return x

######################################
# MobileNet based models (new)
######################################
class MobileNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=1280, feature_input_dim=709, num_classes=3, fusion_dim=256, dropout_rate=0.5):
        """
        Uses a MobileNetV2 backbone for image features and fuses with extra features using attention.
        Note: MobileNetV2 typically outputs 1280-dimensional features.
        """
        super(MobileNetModel_IFOF, self).__init__()
        from torchvision.models import mobilenet_v2
        self.image_cnn = mobilenet_v2(pretrained=True)
        self.image_cnn.classifier = nn.Identity()  # Remove the classifier
        
        self.fusion = FusionAttention(image_dim=image_input_dim,
                                      feature_dim=feature_input_dim,
                                      fusion_dim=fusion_dim)
        
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, images, features):
        # images shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.image_cnn(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Use last output
        
        features = features.view(batch_size, -1)
        
        fused_features = self.fusion(image_features, features)
        
        x = F.relu(self.fc1(fused_features))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class MobileNetModel_IF(nn.Module):
    def __init__(self, num_classes=3):
        super(MobileNetModel_IF, self).__init__()
        from torchvision.models import mobilenet_v2
        self.image_cnn = mobilenet_v2(pretrained=True)
        num_ftrs = self.image_cnn.last_channel  # MobileNetV2's last_channel attribute
        self.image_cnn.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, images):
        x = self.image_cnn(images)
        return x
