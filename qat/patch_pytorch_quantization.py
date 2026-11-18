"""给已安装的 pytorch_quantization 打补丁：让 cuda_ext 变成可选（纯 PyTorch fake-quant）。

背景：本环境 torch 1.10.1+cu113，而 pytorch_quantization 2.1.3 的 cuda_ext 是按别的
torch 版本编译的，import 会报 `undefined symbol: ...SmallVectorBase...`。
但 `tensor_quant.py` 在模块顶层无条件 `from pytorch_quantization import cuda_ext`，
导致 TensorQuantizer 完全不可用（QAT / QDQ 导出都走不下去）。

实际上 fake-quant 有纯 PyTorch 实现（`_tensor_quant` / `legacy_quant_func`），
只要把 cuda_ext 的导入改成可选、并在调用处回退即可。

用法（幂等，可重复执行）:
    python qat/patch_pytorch_quantization.py
"""
import os
import sys

TARGET = "pytorch_quantization/tensor_quant.py"

GUARD = """# cuda_ext 是按别的 torch 版本编译的，import 会失败；降级为纯 PyTorch fake-quant
try:
    from pytorch_quantization import cuda_ext
except ImportError:
    cuda_ext = None"""

MARK = "# [patched by qat/patch_pytorch_quantization.py]"


def main():
    try:
        import pytorch_quantization
    except ImportError as e:
        sys.exit(f"未安装 pytorch_quantization: {e}")
    path = os.path.join(os.path.dirname(pytorch_quantization.__file__), "tensor_quant.py")
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print(f"已打过补丁: {path}")
        return

    # 1) 顶层导入 → 可选（必须是"独占一行的模块级导入"，不能误伤已缩进的行）
    old = "\nfrom pytorch_quantization import cuda_ext\n"
    assert src.count(old) == 1, f"找不到唯一模块级导入（找到 {src.count(old)} 处）"
    src = src.replace(old, "\n" + GUARD + "  " + MARK + "\n", 1)

    # 2) 调用处：cuda_ext 缺失时回退到纯 PyTorch 实现
    old2 = "            try:\n                if amax.numel() == 1:"
    new2 = ("            try:\n"
            "                if cuda_ext is None:\n"
            "                    raise RuntimeError('cuda_ext unavailable, fallback')\n"
            "                if amax.numel() == 1:")
    if old2 in src:
        src = src.replace(old2, new2, 1)
    old3 = "            except (RuntimeError, ValueError) as error:"
    new3 = "            except (RuntimeError, ValueError, AttributeError) as error:"
    if old3 in src:
        src = src.replace(old3, new3, 1)

    open(path, "w", encoding="utf-8").write(src)
    print(f"已打补丁: {path}")
    # 立即验证：能导入 + fake-quant 数值正确
    import torch
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import nn as quant_nn
    q = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8))
    x = torch.tensor([[-2.0, -0.5, 0.0, 0.3, 1.7]])
    q.enable_calib(); q(x); q.load_calib_amax(); q.disable_calib()
    q.use_fb_fake_quant = True
    y = q(x)
    print(f"验证 TensorQuantizer ✓ amax={float(q.amax):.3f} fake-quant={y.flatten().tolist()[:3]}")


if __name__ == "__main__":
    main()
