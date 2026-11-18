import torch


class QuantDescriptor:
    def __init__(self, num_bits=8, axis=None, unsigned=False, calib_method="max", **kw):
        self.num_bits = num_bits
        self.axis = axis
        self.unsigned = unsigned
        self.calib_method = calib_method
        for k, v in kw.items():
            setattr(self, k, v)


def fake_tensor_quant(x, amax=1.0, num_bits=8, **kw):
    return x


def tensor_quant(x, amax=1.0, num_bits=8, **kw):
    return x
