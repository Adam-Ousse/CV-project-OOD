import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=100, track_layers=None):
        super().__init__()
        self.model = resnet18(weights=None)
        
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.feature_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(self.feature_dim, num_classes)
        
        self.track_layers = track_layers or ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
        self._features = {}
        self._hooks = []
        
        self._register_hooks()
        
    def _register_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    if len(output.shape) == 4:
                        self._features[name] = output.mean(dim=[2, 3]).detach()
                    elif len(output.shape) == 2:
                        self._features[name] = output.detach()
                    else:
                        self._features[name] = output.flatten(1).detach()
                elif isinstance(output, (list, tuple)):
                    self._features[name] = output[0].mean(dim=[2, 3]).detach() if len(output[0].shape) == 4 else output[0].detach()
            return hook
        
        layer_map = {
            'conv1': self.model.conv1,
            'bn1': self.model.bn1,
            'relu': self.model.relu,
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4,
            'avgpool': self.model.avgpool,
            'fc': self.model.fc,
        }
        
        for name in self.track_layers:
            if name in layer_map:
                handle = layer_map[name].register_forward_hook(make_hook(name))
                self._hooks.append(handle)
            else:
                print(f"warning: layer {name} not found")
    
    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._features.clear()
    
    def forward(self, x, return_features=False):
        self._features = {}
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.model.fc(features)
        
        if return_features:
            return logits, features, self._features.copy()
        return logits
    
    def get_layer_features(self, x, layer_name):
        self.forward(x)
        return self._features.get(layer_name)
    
    def get_all_layer_features(self, x):
        self.forward(x)
        return self._features.copy()
    
    def get_feature_dims(self):
        dims = {}
        dummy_input = torch.randn(2, 3, 32, 32, device=next(self.parameters()).device)
        
        with torch.no_grad():
            self.forward(dummy_input)
            for name, feat in self._features.items():
                if feat is not None:
                    dims[name] = feat.shape[1]
        
        return dims