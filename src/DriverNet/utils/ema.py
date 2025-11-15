import torch
import torch.nn as nn
import copy

class EMA:
    def __init__(self, model: nn.Module, decay: float, warmup_steps: int = 0):
        self.model = model
        self.teacher = self._create_teacher()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.global_step = 0

    def _create_teacher(self) -> nn.Module:
        teacher = copy.deepcopy(self.model)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return teacher

    @torch.no_grad()
    def update(self):
        self.global_step += 1
        if self.global_step < self.warmup_steps:
            self.teacher.load_state_dict(self.model.state_dict())
            return

        m = self.decay
        for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

        for t_buf, s_buf in zip(self.teacher.buffers(), self.model.buffers()):
            t_buf.data.copy_(s_buf.data)

    def __call__(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)
