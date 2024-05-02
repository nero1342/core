from typing import Dict
import logging 
import torch 
import math
from omegaconf import DictConfig 
import numpy as np 
from torch.cuda.amp import autocast

from core.utils import comm
from core.utils.events import EventStorage
from .builder import TrainerBuilder

logger = logging.getLogger(__name__)

class TrainerBase(TrainerBuilder):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.use_amp = False
        self.clip_grad_norm = 0.05
        
        if cfg.solver.get("use_amp"):
            self.use_amp = True
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()

        self.start_iter = 0
    
    def train(self):
        self.iter = self.start_iter
        max_iter = self.cfg.solver.max_iter

        dataloader = self.build_train_dataloader()
        logger.info("Starting training from iteration {}".format(self.start_iter))

        # determine max epoch
        total_epoch = math.ceil(max_iter / len(dataloader))
        current_epoch = self.iter // len(dataloader)
        logger.info(f'We will approximately train with {total_epoch} epochs.')

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                while self.iter < max_iter:
                    for data in dataloader:
                        self.before_step()
                        losses = self.run_step(data, self.iter)
                        self.backward(losses)
                        self.after_step()
                        self.iter += 1

                        if self.iter >= max_iter:
                            break
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
    
    def run_step(self, data, cur_iter=0):
        # logger.info(f"Running iter: {cur_iter}...")
        with autocast(self.use_amp):
            output = self.model(data)
        
            # Dummy
            output = torch.nn.functional.cross_entropy(output, torch.rand(4, 1000, device = 'cuda'))
            output = {"losses": output}

            loss_dict = output["losses"]
            if isinstance(loss_dict, torch.Tensor):
                loss_dict = {"total_loss": loss_dict}

        self.write_metrics(loss_dict, cur_iter)

        # Log images or do some other stuff using data and output

        losses = sum(loss_dict.values())
        return losses 
        
    def write_metrics(self, loss_dict: Dict[str, torch.Tensor], cur_iter: int = 0):
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

            self.storage.put_scalar(
                "total_loss", total_losses_reduced, cur_iter=cur_iter
            )
            if len(metrics_dict) > 1:
                self.storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

            self.storage.put_scalar(
                "lr", self.optimizer.param_groups[-1]["lr"], smoothing_hint=False
            )
    
    def backward(self, losses: torch.Tensor):
        """
            Backward - Optimizer - Scheduler 
        """
        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(losses).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
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
        return [torch.rand(4, 3, 244, 244) for _ in range(10)]

    def before_train(self):
        self.storage.iter = self.iter
        for hook in self.hooks:
            hook.before_train()
    
    def before_step(self):
        self.storage.iter = self.iter
        for hook in self.hooks:
            hook.before_step()
    
    def after_step(self):
        self.storage.iter = self.iter
        for hook in self.hooks:
            hook.after_step()
    
    def after_train(self):
        self.storage.iter = self.iter
        for hook in self.hooks:
            hook.after_train()

 