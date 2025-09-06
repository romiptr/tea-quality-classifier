import torch
import torch.nn as nn
from timm import create_model

class ImageClassification(nn.Module):
    def __init__(self, 
                 num_classes,
                 backbone="resnet50", 
                 pretrained=True):
        super().__init__()

        # timm wrapper to get backbone network removes final pooling and classification layers
        self.encoder = create_model(
            backbone, pretrained=pretrained, 
            num_classes=0, global_pool=''
        )

        # detect feature size dynamically for timm models
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.encoder(dummy_input)
            num_features = dummy_features.shape[1]

        # classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        output = self.head(features)

        return output

if __name__ == '__main__':

    # model = ImageClassification(backbone="mobilenetv3_small_100", num_classes=3)
    model = ImageClassification(backbone="starnet_s1.in1k", num_classes=3)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        predictions = model(dummy_input)

    print(f"Output shape: {predictions.shape}")