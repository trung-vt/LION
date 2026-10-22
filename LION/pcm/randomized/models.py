"""Model factories for bounded randomized-noise pilot experiments."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml
from deepinv.models import DRUNet

from LION.config.sapm_config import make_sapm_model_config
from LION.models.AdaptiveTGVPCMNetwork import AdaptiveTGVPCMNetwork
from LION.models.AdaptiveTVPCMNetwork import AdaptiveTVPCMNetwork
from LION.models.ParameterMapNetwork import ParameterMapNetwork
from LION.models.fully_learned.DWTDenoiseNet import load_dwtdnet
from LION.models.unet import UNet
from LION.reconstructors.GSPnP import GSDenoiser


DENOISERS = ("dwtdnet", "gs_dwtdnet", "drunet", "gs_drunet")
REGULARIZERS = ("utv_supervised", "utgv_supervised", "utv_ssdu", "utgv_ssdu")


def _smooth_activations(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.ReLU):
            setattr(module, name, torch.nn.Softplus(beta=1.0))
        else:
            _smooth_activations(child)


def build_denoiser(name: str, side: int, device: torch.device) -> torch.nn.Module:
    if name not in DENOISERS:
        raise ValueError(name)
    if "dwtdnet" in name:
        model = load_dwtdnet("tiny", side, device=device)
        # Avoid clipping the potential network or restricting heavy-noise outputs.
        model.clamp_output = False
    else:
        model = DRUNet(in_channels=1, out_channels=1, pretrained=None, device=device)
        if name.startswith("gs_"):
            _smooth_activations(model)
    if name.startswith("gs_"):
        model = GSDenoiser(model)
    return model.to(device)


def denoise(
    model: torch.nn.Module, x: torch.Tensor, sigma: float, *, training: bool
) -> torch.Tensor:
    if isinstance(model, GSDenoiser):
        return model(x, sigma=sigma, set_reset_grad=not training)
    return model(x, sigma)


def build_regularizer(
    name: str,
    side: int,
    iterations: int,
    device: torch.device,
) -> torch.nn.Module:
    if name not in REGULARIZERS:
        raise ValueError(name)
    is_tgv = name.startswith("utgv")
    root = Path(__file__).resolve().parents[3]
    raw = copy.deepcopy(
        yaml.safe_load((root / "config/pcm/pcm_tv_supervised.yaml").read_text())[
            "model"
        ]
    )
    raw.update(
        backend="direct",
        n_iters=iterations,
        input_image="adjoint",
        initial_image="adjoint",
    )
    raw["pdhg"].update(
        regularisation="TGV" if is_tgv else "TV",
        primal_stepsize=0.2,
        dual_stepsize=0.2,
    )
    raw["cnn"]["out_channels"] = 2 if is_tgv else 1
    config = make_sapm_model_config(raw)
    cnn = config.cnn
    map_maker = UNet(
        dim=cnn.dim,
        padding_mode=cnn.padding_mode,
        n_filters=cnn.init_filters,
        n_enc_stages=cnn.n_blocks,
        n_ch_in=cnn.in_channels,
        n_ch_out=cnn.out_channels,
        n_convs_per_stage=cnn.n_convs_per_block,
        kernel_size=cnn.kernel_size,
        bias=cnn.bias,
        pooling_kernel_size=cnn.downsampling_kernel,
    )
    # A single shared scale key lets the map model generalize across sampled operators.
    map_model = ParameterMapNetwork(
        map_maker,
        keys=(0,),
        initial_scale_factor_raw=-4.0,
        scale_factor_softplus_beta=1.0,
        lambda_softplus_beta=1.0,
    )
    cls = AdaptiveTGVPCMNetwork if is_tgv else AdaptiveTVPCMNetwork
    return cls(config, lambda_map_network=map_model).to(device)


def regularized_reconstruction(
    model: torch.nn.Module, op, y: torch.Tensor
) -> torch.Tensor:
    initial = op.adjoint(y)
    return model(
        input_image=initial,
        initial_image=initial,
        measurement=y,
        key=0,
        forward_op=op,
        use_checkpoint=False,
        use_tqdm=False,
    )
