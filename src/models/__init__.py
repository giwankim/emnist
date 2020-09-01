from models.baseline import Baseline
from models.spinalvgg import SpinalVGG
from models.efficientnet import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    EfficientNetB8,
    EfficientNetL2,
)
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def get_model(model_name):
    if model_name == "baseline":
        return Baseline()
    elif model_name == "spinalvgg":
        return SpinalVGG()
    elif model_name == "efficientnet-b0":
        return EfficientNetB0()
    elif model_name == "efficientnet-b1":
        return EfficientNetB1()
    elif model_name == "efficientnet-b2":
        return EfficientNetB2()
    elif model_name == "efficientnet-b3":
        return EfficientNetB3()
    elif model_name == "efficientnet-b4":
        return EfficientNetB4()
    elif model_name == "efficientnet-b5":
        return EfficientNetB5()
    elif model_name == "efficientnet-b6":
        return EfficientNetB6()
    elif model_name == "efficientnet-b7":
        return EfficientNetB7()
    elif model_name == "efficientnet-b8":
        return EfficientNetB8()
    elif model_name == "efficientnet-l2":
        return EfficientNetL2()
    elif model_name == "resnet18":
        return ResNet18()
    elif model_name == "resnet34":
        return ResNet34()
    elif model_name == "resnet50":
        return ResNet50()
    elif model_name == "resnet101":
        return ResNet101()
    elif model_name == "resnet152":
        return ResNet152()
    else:
        raise RuntimeError(f"Unknown model name: {model_name}")
