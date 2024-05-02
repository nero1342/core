import torch 
from torch import nn

from .base import HookBase

class PeriodCheckpointer(HookBase):
    def __init__(self, model: nn.Module, period: int):
        super().__init__()