from torch.nn import Module
from typing import Callable, get_args
from .alexnet import load_alexnet, alexnet_model_names
from .resnet import load_resnet, resnet_model_names
from .vgg import load_vgg, vgg_model_names
from .googlenet import load_googlenet, googlenet_model_names
from .vit import load_vit, vit_model_names

MODEL_OPTIONS: dict[str, Callable[..., Module]] = {
    **{
        f"alexnet{m}": (lambda m=m, **kw: load_alexnet(m, **kw))
        for m in get_args(alexnet_model_names)
    },
    **{
        f"googlenet{m}": (lambda m=m, **kw: load_googlenet(m, **kw))
        for m in get_args(googlenet_model_names)
    },
    **{
        f"resnet{m}": (lambda m=m, **kw: load_resnet(m, **kw))
        for m in get_args(resnet_model_names)
    },
    **{
        f"vgg{m}": (lambda m=m, **kw: load_vgg(m, **kw))
        for m in get_args(vgg_model_names)
    },
    **{
        f"vit_{m}": (lambda m=m, **kw: load_vit(m, **kw))
        for m in get_args(vit_model_names)
    },
}

MODEL_NAMES = list(MODEL_OPTIONS.keys())
