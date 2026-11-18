"""pytorch_quantization 的最小替身。

原包（NVIDIA）的 cuda_ext 是按别的 torch 版本编译的，在本环境 import 会报
undefined symbol: _ScaleTorchVersion...。fp16 导出用不到真正的量化算子，
这里提供最小 API 让 qat/lean 能 import，量化器全部退化为恒等映射。
"""
