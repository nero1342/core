import logging 
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def get_parameter_groups(cfg: DictConfig, model: torch.nn.Module):
    """
    Assign different weight decays and learning rates to different parameters.
    Returns a parameter group which can be passed to the optimizer.
    """
    weight_decay = cfg.weight_decay
    # embed_weight_decay = stage_cfg.embed_weight_decay
    backbone_lr_multiplier = cfg.backbone_lr_multiplier
    base_lr = cfg.base_lr

    memo: Set[torch.nn.parameter.Parameter] = set()

    backbone_params = []
    embed_params = []
    other_params = []


    for name, param in model.named_parameters():
        # logger.info(name)
        if not param.requires_grad:
            continue
        # Avoid duplicating parameters
        if param in memo:
            continue
        memo.add(param)
    
        if name.startswith("backbone."):
            backbone_params.append(param)
            logger.info(f"{name} counted as a backbone parameter.")
            continue

        if name.startswith("fc."):
            embed_params.append(param)
            logger.info(f"{name} counted as a embed parameter.")
            continue 

        other_params.append(param)

    parameter_groups = [
        {
            "params": backbone_params,
            "lr": base_lr * backbone_lr_multiplier,
            "weight_decay": weight_decay
        },
        {
            "params": embed_params,
            "lr": base_lr,
            "weight_decay": weight_decay
        },
        {
            "params": other_params,
            "lr": base_lr,
            "weight_decay": weight_decay
        },
    ]

    return parameter_groups
                