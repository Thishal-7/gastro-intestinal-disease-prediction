import torch
import torch.nn as nn
from torchvision import models
import timm
import joblib   # ✅ NEW (for lab model)

# =========================
# 🔹 IMAGE MODELS
# =========================

class InceptionResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(InceptionResNetModel, self).__init__()
        self.inception_resnet = timm.create_model('inception_resnet_v2', pretrained=True)
        
        num_features = self.inception_resnet.classif.in_features
        self.inception_resnet.classif = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.inception_resnet(x)


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.resnet = models.resnet50(weights=None)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-4]:
            param.requires_grad = False
            
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class DenseNet121Model(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121Model, self).__init__()
        self.densenet = models.densenet121(weights=None)
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.densenet(x)
    
    # For Grad-CAM
    def get_cam_layer(self):
        return self.densenet.features.denseblock4.denselayer16


# =========================
# 🔹 ENSEMBLE MODEL
# =========================

class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
    
    def forward(self, x):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            outputs.append(model(x) * weight)
        return torch.sum(torch.stack(outputs), dim=0)


# =========================
# 🔹 LOAD IMAGE MODEL (NEW)
# =========================

def load_ensemble_model(num_classes=3):
    model1 = InceptionResNetModel(num_classes)
    model2 = Resnet50(num_classes)
    model3 = DenseNet121Model(num_classes)

    ensemble = EnsembleModel([model1, model2, model3])

    # Load trained weights (IMPORTANT: update path)
    ensemble.load_state_dict(torch.load("ensemble_model.pth", map_location=torch.device('cpu')))
    ensemble.eval()

    return ensemble


# =========================
# 🔹 LAB MODEL (NEW)
# =========================

def load_lab_model():
    """
    Loads ML model trained on lab data
    """
    model = joblib.load("lab_model.pkl")
    return model

import joblib

def load_lab_model():
    model, encoder, columns = joblib.load("lab_model.pkl")
    return model, encoder, columns
import joblib

def load_lab_model():
    model, encoder, columns = joblib.load("lab_model.pkl")
    return model, encoder, columns
