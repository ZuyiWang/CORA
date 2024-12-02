from .fast_detr import build
from .ov_dino.dino import build_dino


def build_model(args):
    if args.use_dino:
        return build_dino(args)
    return build(args)
