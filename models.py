import torch.nn as nn
from torchvision import models


def get_resnets(num_out_classes, pretrained, train_config):
    contexts = {
        "resnet18": models.resnet18(pretrained=pretrained),
        "resnet50": models.resnet50(pretrained=pretrained),
        "resnet101": models.resnet101(pretrained=pretrained),
        "resnext50_32x4d": models.resnext50_32x4d(pretrained=pretrained),
        "resnext101_32x8d": models.resnext101_32x8d(pretrained=pretrained),
    }
    base_model = contexts[train_config["models"]["name"]]
    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, num_out_classes)

    return base_model


if __name__ == "__main__":
    model = get_resnets(196, True)
    print(model)
