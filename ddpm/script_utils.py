import argparse
import torchvision
import torch.nn.functional as F

from .unet import UNet
from .diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data

def get_transform():
    # class Normalize(object):
    #     def __call__(self, tensor):
    #         return tensor.float() / 255.0
        
    return torchvision.transforms.Compose([
        # Normalize(),
        torchvision.transforms.ToTensor(),
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=False,

        base_channels=32,
        num_groups=8,
        channel_mults=(1, 1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(),

        ema_decay=0.9999,
        ema_update_rate=1,
    )

    return defaults


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    model = UNet(
        img_channels=3,

        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        num_groups=args.num_groups,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,
        initial_pad=0,
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model, (480, 640), 3, 10,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
    )

    return diffusion