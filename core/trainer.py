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

class TrainerBase:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        model = self.build_model(cfg.model)
        self.optimizer = self.build_optimizer(cfg.solver, model)
        self.scheduler = self.build_scheduler(cfg.solver, self.optimizer)

        self.model = create_ddp_model(model)

        self.build_hooks(cfg)
    

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

        ]
        if comm.is_main_process():
            ret.append(hooks.PeriodCheckpointer(self.model, cfg.solver.checkpoint_period))
        
        ret.append(hooks.EvalHook(cfg.test.eval_period))

        if comm.is_main_process():
            ret.append(hooks.PeriodWriter(build_writers(cfg.writer), period=cfg.writer.period))
    
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

class Trainer(TrainseBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        model = self.build_model(cfg.model)
        self.optimizer = self.build_optimizer(cfg.solver, model)
        self.scheduler = self.build_scheduler(cfg.solver, self.optimizer)

        self.model = create_ddp_model(model)

        self.start_iter = 0

        self.cfg = cfg

        self.build_hooks(cfg)

    def train(self):
        curr_iter = start_iter = self.start_iter
        max_iter = self.cfg.solver.max_iter


        dataloader = self.build_train_dataloader()
        logger.info("Starting training from iteration {}".format(start_iter))

        # determine max epoch
        total_epoch = math.ceil(total_iterations / len(dataloader))
        current_epoch = curr_iter // len(dataloader)
        logger.info(f'We will approximately train with {total_epoch} epochs.')

        with EventStorage(start_iter) as self.storage:
            try:
                while curr_iter < max_iter:
                    for data in dataloader:
                        losses = self.run_step(data, curr_iter)
                        self.backward(losses)
                        self.after_step()
                        curr_iter += 1

                        if curr_iter >= max_iter:
                            break
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
    
    def run_step(self, data, cur_iter=0):
        output = self.model(data)
        loss_dict = output["losses"]

        if isinstance(loss_dict, torch.Tensor):
            loss_dict = {"total_loss": loss_dict}

        self.write_metrics(loss_dict)

        # Log images or do some other stuff using data and output

        losses = sum(loss_dict.values())
        return losses 
        
    def write_metrics(self, loss_dict: Dict[str, torch.Tensor]):
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar(
                "total_loss", total_losses_reduced, cur_iter=cur_iter
            )
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)
    
    def backward(self, losses: torch.Tensor):
        """
            Backward - Optimizer - Scheduler 
        """
        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(losses).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.cutie.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            losses.backward()
            if self.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

        self.scheduler.step()

    def build_train_dataloader(self):
        cfg = self.cfg


 