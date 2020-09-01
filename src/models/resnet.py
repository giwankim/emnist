import torch.nn as nn
import pretrainedmodels


def ResNet18(pretrained=None):
    model = pretrainedmodels.__dict__["resnet18"](pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    in_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(in_feats, 10)
    return model


def ResNet34(pretrained=None):
    model = pretrainedmodels.__dict__["resnet34"](pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    in_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(in_feats, 10)
    return model


def ResNet50(pretrained=None):
    model = pretrainedmodels.__dict__["resnet50"](pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    in_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(in_feats, 10)
    return model


def ResNet101(pretrained=None):
    model = pretrainedmodels.__dict__["resnet101"](pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    in_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(in_feats, 10)
    return model


def ResNet152(pretrained=None):
    model = pretrainedmodels.__dict__["resnet152"](pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    in_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(in_feats, 10)
    return model
