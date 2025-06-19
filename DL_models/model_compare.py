import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import densenet121
from torchvision.models import mobilenet_v2

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

class ResNetModel_IF(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5, freeze_until_layer=5):
        super(ResNetModel_IF, self).__init__()
        self.image_cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace final FC layer with an identity to extract features
        in_features = self.image_cnn.fc.in_features
        self.image_cnn.fc = nn.Identity()
        
        # Freeze early layers to retain low‑level features
        child_counter = 0
        for child in self.image_cnn.children():
            child_counter += 1
            if child_counter < freeze_until_layer:
                for param in child.parameters():
                    param.requires_grad = False
        
        # Build a new classifier head
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, images):
        # images shape: (batch_size, channels, height, width)
        features = self.image_cnn(images)
        x = F.relu(self.bn1(self.fc1(features)))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


class ResNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=512, feature_input_dim=709, num_classes=3, 
                 fusion_dim=256, dropout_rate=0.5, freeze_until_layer=5):
        """
        Uses a ResNet18 backbone for image features and fuses with extra features using attention.
        Freezes early layers and uses an enhanced classifier head.
        """
        super(ResNetModel_IFOF, self).__init__()
        self.image_cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.image_cnn.fc.in_features
        self.image_cnn.fc = nn.Identity()  # Remove the final FC layer
        
        # Freeze early layers to preserve low-level features
        child_counter = 0
        for child in self.image_cnn.children():
            child_counter += 1
            if child_counter < freeze_until_layer:
                for param in child.parameters():
                    param.requires_grad = False
        
        # Fusion module remains unchanged
        self.fusion = FusionAttention(image_dim=image_input_dim,
                                      feature_dim=feature_input_dim,
                                      fusion_dim=fusion_dim)
        
        # Enhanced classifier head: increased hidden dimensions with BatchNorm and Dropout
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, images, features):
        # images shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.image_cnn(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Using the last output
        
        features = features.view(batch_size, -1)  # Flatten extra features
        
        # Fuse image and extra features
        fused_features = self.fusion(image_features, features)
        
        x = F.relu(self.bn1(self.fc1(fused_features)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x



class EfficientNetModel_IF(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5, freeze_features=True):
        super(EfficientNetModel_IF, self).__init__()
        # Load a pretrained EfficientNet-B0
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Get the number of features from the classifier
        in_features = self.efficientnet.classifier[1].in_features
        # Remove the default classifier
        self.efficientnet.classifier = nn.Identity()
        
        # Optionally freeze the feature extractor to preserve low-level features
        if freeze_features:
            for param in self.efficientnet.features.parameters():
                param.requires_grad = False
        
        # Build a new classifier head
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, images):
        # images shape: (batch_size, channels, height, width)
        features = self.efficientnet(images)
        x = F.relu(self.bn1(self.fc1(features)))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


class EfficientNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=1280, feature_input_dim=709, num_classes=3, 
                 fusion_dim=256, dropout_rate=0.5, freeze_features=True):
        """
        Uses an EfficientNet-B0 backbone for image features and fuses with extra features using attention.
        Optionally freezes the feature extractor.
        """
        super(EfficientNetModel_IFOF, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()  # Remove the final classifier
        
        if freeze_features:
            for param in self.efficientnet.features.parameters():
                param.requires_grad = False
        
        self.fusion = FusionAttention(image_dim=image_input_dim,
                                      feature_dim=feature_input_dim,
                                      fusion_dim=fusion_dim)
        
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, images, features):
        # images shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.efficientnet(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Last frame's features

        features = features.view(batch_size, -1)  # Flatten extra features
        
        fused_features = self.fusion(image_features, features)
        
        x = F.relu(self.bn1(self.fc1(fused_features)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class DenseNetModel_IF(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5, freeze_features=True):
        super(DenseNetModel_IF, self).__init__()
        self.image_cnn = densenet121(pretrained=True)
        in_features = self.image_cnn.classifier.in_features
        
        # Optionally freeze feature extractor
        if freeze_features:
            for param in self.image_cnn.features.parameters():
                param.requires_grad = False
        
        # Replace the classifier with a new one
        self.image_cnn.classifier = nn.Identity()
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, images):
        # images shape: (batch_size, channels, height, width)
        features = self.image_cnn(images)
        x = F.relu(self.bn1(self.fc1(features)))
        x = self.dropout(x)
        out = self.fc2(x)
        return out


class DenseNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=1024, feature_input_dim=709, num_classes=3, 
                 fusion_dim=256, dropout_rate=0.5, freeze_features=True):
        """
        Uses a DenseNet121 backbone for image features and fuses with extra features using attention.
        Optionally freezes the DenseNet feature extractor.
        """
        super(DenseNetModel_IFOF, self).__init__()
        self.image_cnn = densenet121(pretrained=True)
        self.image_cnn.classifier = nn.Identity()  # Remove the classifier
        
        if freeze_features:
            for param in self.image_cnn.features.parameters():
                param.requires_grad = False
        
        self.fusion = FusionAttention(image_dim=image_input_dim,
                                      feature_dim=feature_input_dim,
                                      fusion_dim=fusion_dim)
        
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, images, features):
        # images shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.image_cnn(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Last frame's features
        
        features = features.view(batch_size, -1)  # Flatten extra features
        
        fused_features = self.fusion(image_features, features)
        
        x = F.relu(self.bn1(self.fc1(fused_features)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class MobileNetModel_IF(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5, freeze_features=True):
        super(MobileNetModel_IF, self).__init__()
        self.image_cnn = mobilenet_v2(pretrained=True)
        # Optionally freeze the feature extractor
        if freeze_features:
            for param in self.image_cnn.features.parameters():
                param.requires_grad = False
        
        num_ftrs = self.image_cnn.last_channel  # Typically 1280
        self.image_cnn.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, images):
        # images shape: (batch_size, channels, height, width)
        x = self.image_cnn(images)
        return x


class MobileNetModel_IFOF(nn.Module):
    def __init__(self, image_input_dim=1280, feature_input_dim=709, num_classes=3, 
                 fusion_dim=256, dropout_rate=0.5, freeze_features=True):
        """
        Uses a MobileNetV2 backbone for image features and fuses with extra features using attention.
        Optionally freezes the MobileNet feature extractor.
        """
        super(MobileNetModel_IFOF, self).__init__()
        self.image_cnn = mobilenet_v2(pretrained=True)
        self.image_cnn.classifier = nn.Identity()  # Remove the classifier
        
        if freeze_features:
            for param in self.image_cnn.features.parameters():
                param.requires_grad = False
        
        self.fusion = FusionAttention(image_dim=image_input_dim,
                                      feature_dim=feature_input_dim,
                                      fusion_dim=fusion_dim)
        
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, images, features):
        # images shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.image_cnn(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Last frame's features
        
        features = features.view(batch_size, -1)
        
        fused_features = self.fusion(image_features, features)
        
        x = F.relu(self.bn1(self.fc1(fused_features)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
