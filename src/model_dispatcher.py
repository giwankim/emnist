import models


def get_model(model_name):
    if model_name == "baseline":
        return models.Model()
    elif model_name == "spinalvgg":
        return models.SpinalVGG()
    else:
        raise RuntimeError("Not a valid model name!")
