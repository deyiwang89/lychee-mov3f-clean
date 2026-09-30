import torch
import torch.nn as nn

try:
    from torchvision.models import mobilenet_v3_small as tv_mobilenet_v3_small
    from torchvision.models import mobilenet_v3_large as tv_mobilenet_v3_large
    from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
except ImportError:
    tv_mobilenet_v3_small = None
    tv_mobilenet_v3_large = None

class MobileNetV3Wrapper(nn.Module):
    def __init__(self, base_model, num_classes=1000):
        super(MobileNetV3Wrapper, self).__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        
        # If the target number of classes is different, we replace the last layer
        if num_classes != 1000:
            in_features = self.classifier[-1].in_features
            self.classifier[-1] = nn.Linear(in_features, num_classes)
            
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
            
    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

def mobilenet_v3_small(pretrained=False, progress=True, num_classes=1000):
    if tv_mobilenet_v3_small is None:
        raise RuntimeError("torchvision is required for MobileNetV3")
        
    if pretrained:
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            base_model = tv_mobilenet_v3_small(weights=weights, progress=progress)
        except:
            base_model = tv_mobilenet_v3_small(pretrained=True, progress=progress)
    else:
        base_model = tv_mobilenet_v3_small(pretrained=False, progress=progress)
        
    model = MobileNetV3Wrapper(base_model, num_classes=num_classes)
    return model

def mobilenet_v3_large(pretrained=False, progress=True, num_classes=1000):
    if tv_mobilenet_v3_large is None:
        raise RuntimeError("torchvision is required for MobileNetV3")
        
    if pretrained:
        try:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            base_model = tv_mobilenet_v3_large(weights=weights, progress=progress)
        except:
            base_model = tv_mobilenet_v3_large(pretrained=True, progress=progress)
    else:
        base_model = tv_mobilenet_v3_large(pretrained=False, progress=progress)
        
    model = MobileNetV3Wrapper(base_model, num_classes=num_classes)
    return model
