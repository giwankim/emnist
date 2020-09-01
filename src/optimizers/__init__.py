import torch
import adamp

from .radam import RAdam, RAdam_4step


def get_optimizer(optim_name, model, lr, weight_decay=1e-4, skip_list=["bias", "bn"]):
    params = []
    exclude_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        elif any(skip in name for skip in skip_list):
            exclude_params.append(param)
        else:
            params.append(param)
    optim_params = [
        {"params": params, "weight_decay": weight_decay},
        {"params": exclude_params, "weight_decay": 0.0},
    ]
    
    if optim_name == "adam":
        optimizer = torch.optim.Adam(optim_params, lr=lr, betas=(0.9, 0.999))
    elif optim_name == "radam":
        optimizer = RAdam(optim_params, lr=lr, betas=(0.9, 0.999))
    elif optim_name == "radam4s":
        optimizer = RAdam_4step(optim_params, lr=lr, betas=(0.9, 0.999))
    elif optim_name == "adamp":
        optimizer = adamp.AdamP(optim_params, lr=lr, betas=(0.9, 0.999))
    elif optim_name == "sgdp":
        optimizer = adamp.SGDP(optim_params, lr=lr, momentum=0.9, nesterov=True)
    else:
        raise RuntimeError(f"Unknown optimizer name: {optim_name}")
        
    return optimizer
