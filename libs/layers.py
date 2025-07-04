import torch
import torch.nn as nn


class BaseModel(nn.Module):
    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def init_weights(self):
        self.apply(BaseModel._init_weights)

    def requires_grad(self, requires_grad: bool):
        for p in self.parameters():
            p.requires_grad = requires_grad
