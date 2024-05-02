from .base import HookBase

class EvalHook(HookBase):
    def __init__(self, period: int):
        super().__init__()