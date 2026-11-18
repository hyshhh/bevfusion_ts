import torch

from .modules._utils import QuantInputMixin, QuantMixin  # noqa: F401
from ..tensor_quant import QuantDescriptor


class TensorQuantizer(torch.nn.Module):
    """恒等量化器：fp16 导出时不做任何量化。"""

    use_fb_fake_quant = False

    def __init__(self, quant_desc=None, **kw):
        super().__init__()
        self._calibrator = None
        self.register_buffer("amax", torch.tensor(1.0))
        self._disabled = True

    def forward(self, x, *a, **kw):
        return x

    def disable(self):
        self._disabled = True

    def enable(self):
        self._disabled = False


class _Dummy(torch.nn.Module):
    def forward(self, x, *a, **kw):
        return x


QuantConv2d = _Dummy
QuantLinear = _Dummy
QuantConv3d = _Dummy
