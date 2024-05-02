import os 
import logging
import torch 
import hydra
from torch import distributed
from omegaconf import OmegaConf, DictConfig

from core.engine.train_loop import TrainerBase as Trainer
from core.utils.utils import extras

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

from core.utils import comm

logger = logging.getLogger(__name__)

def distributed_setup():
    distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    logger.info(f'Initialized: local_rank={comm.get_rank()}, world_size={comm.get_world_size()}')
    logger.info(f"Number of GPUs: {world_size} with current local_rank={local_rank}")

def logger_setup():
    # Logging in master only
    if local_rank != 0:
        logging.disable(logging.CRITICAL)


@hydra.main(version_base='1.3.2', config_path='config', config_name='default')
def train(cfg: DictConfig):
    # init setup
    logger_setup()
    distributed_setup()
    
    extras(cfg)

    trainer = Trainer(cfg)
    trainer.train()

    # clean-up
    distributed.destroy_process_group()

if __name__ == "__main__":
    train()