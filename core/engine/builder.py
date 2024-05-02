import logging 
from omegaconf import DictConfig 
import torch
from torch import optim

from torch.nn.parallel import DistributedDataParallel

from core.model.utils.parameter_groups import get_parameter_groups
from core.utils.model_summary import model_summary
from core.utils.events import EventStorage
from core.utils import comm 
from core.writer import build_writers

logger = logging.getLogger(__name__)

class TrainerBuilder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        model = self.build_model(cfg.model)
        self.optimizer = self.build_optimizer(cfg.solver, model)
        self.scheduler = self.build_scheduler(cfg.solver, self.optimizer)

        self.model = create_ddp_model(model)

        self.hooks = self.build_hooks(cfg)

    def resume_or_load(self, resume: bool = False):
        raise NotImplementedError

    @classmethod
    def build_model(cls, cfg: DictConfig):
        logger.info("Building model...")
        from torchvision import models
        model = models.resnet101().cuda()

        logger.info(model_summary(model))
        return model 

    @classmethod
    def build_optimizer(cls, cfg: DictConfig, model: torch.nn.Module) -> optim.Optimizer:
        param_groups = get_parameter_groups(cfg, model)
        
        optimizer_type = cfg.optimizer
        if optimizer_type == "SGD":
            optimizer = optim.SGD(param_groups, lr=cfg.base_lr, momentum=cfg.momentum)
        elif optimizer_type == "AdamW":
            optimizer = optim.AdamW(param_groups, lr=cfg.base_lr)
        else:
            raise NotImplementedError(f"No optimizer type {optimizer_type}") 

        return optimizer

    @classmethod
    def build_scheduler(cls, cfg: DictConfig, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
        if cfg.lr_scheduler == 'poly':
            total_iterations = cfg.max_iter
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (1 - (x / total_iterations)) ** 0.9)
        elif cfg.lr_scheduler == 'step':
            return optim.lr_scheduler.MultiStepLR(optimizer, cfg.steps, cfg.scheduler_gamma)
        else:
            raise NotImplementedError
    
    def build_hooks(self, cfg):
        from core import hooks
        ret = [
            hooks.IterationTimer()
        ]
        if comm.is_main_process():
            ret.append(hooks.PeriodCheckpointer(self.model, cfg.solver.checkpoint_period))
        
        ret.append(hooks.EvalHook(cfg.test.eval_period))

        if comm.is_main_process():
            ret.append(hooks.PeriodWriter(build_writers(cfg.writer), period=cfg.writer.period))

        return ret
def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model

    local_rank = comm.get_rank()
    ddp = DistributedDataParallel(model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False
    )

    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp
