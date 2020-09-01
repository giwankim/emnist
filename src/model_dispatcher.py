import models
import pretrainedmodels


def get_model(model_name):
    if model_name == "baseline":
        return models.Model()
    elif model_name == "spinalvgg":
        return models.SpinalVGG()
    else:
        raise RuntimeError("Not a valid model name!")
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

def EfficientNetB1():
    model = EfficientNet.from_name("efficientnet-b1", image_size=28, num_classes=10)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 32,
        kernel_size=(3, 3),
        stride=(1,1),
        bias=False,
        image_size=(28, 28))
    return model

def EfficientNetB2():
    model = EfficientNet.from_name("efficientnet-b2", image_size=28, num_classes=10)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 32,
        kernel_size=(3,3),
        stride=(1,1),
        bias=False,
        image_size=(28, 28)
    )
    return model

def EfficientNetB3():
    model = EfficientNet.from_name("efficientnet-b3", image_size=28, num_classes=10)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 32,
        kernel_size=(3, 3),
        stride=(1, 1),
        bias=False,
        image_size=(28, 28)
    )
    return model

def EfficientNetB4():
    model = EfficientNet.from_name("efficientnet-b4", image_size=28, num_classes=10)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 48,
        kernel_size=(3, 3),
        stride=(1, 1),
        bias=False,
        image_size=(28, 28)
    )
    return model

def EfficientNetB5():
    model = EfficientNet.from_name("efficientnet-b5", image_size=28, num_classes=10)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 32, kernel_size=3, stride=1, bias=False, image_size=(28, 28)
    )
    return model

def EfficientNetB6():
    model = EfficientNet.from_name("efficientnet-b6", image_size=28, num_classes=10)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 56,
        kernel_size=(3, 3),
        stride=(1, 1),
        bias=False,
        image_size=(28, 28)
    )
    return model

def EfficientNetB7():
    model = EfficientNet.from_name("efficientnet-b7", image_size=28, num_classes=10)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 56,
        kernel_size=(3, 3),
        stride=(1, 1),
        bias=False,
        image_size=(28, 28)
    )
    return model

def EfficientNetB8():
    model = EfficientNet.from_name("efficientnet-b8", image_size=28, num_classes=10)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 56,
        kernel_size=(3, 3),
        stride=(1, 1),
        bias=False,
        image_size=(28, 28)
    )
    return model

def EfficientNetL2():
    model = EfficientNet.from_name("efficientnet-l2", image_size=28, num_classes=10)
    model._conv_stem = Conv2dStaticSamePadding(
        3, 56,
        kernel_size=(3, 3),
        stride=(1, 1),
        bias=False,
        image_size=(28, 28)
    )
    return model