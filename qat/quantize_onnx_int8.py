"""把导出的 fp32 ONNX 转成 INT8（QDQ）ONNX —— 不依赖 pytorch_quantization。

思路：用 onnxruntime 的静态量化（PTQ）：
  1) 读 fp32 ONNX 的输入签名，生成/加载校准数据；
  2) 跑 CalibrationDataReader 统计每层激活范围；
  3) 输出带 QuantizeLinear/DequantizeLinear 的 int8 图。

用法:
    # 单个文件（随机校准）
    python qat/quantize_onnx_int8.py qat/onnx_int8/fuser.onnx
    # 指定输出与校准样本数
    python qat/quantize_onnx_int8.py qat/onnx_int8/fuser.onnx \
        --out qat/onnx_int8/fuser.int8.onnx --calib 16
    # 用真实校准数据（npz: 键名 = 输入名, 值 = ndarray，第 0 维是样本）
    python qat/quantize_onnx_int8.py x.onnx --calib-npz calib.npz

注意：含自定义算子（spconv 的 lidar.backbone）无法被 onnxruntime 量化，会跳过。
"""
import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
    quantize_dynamic,
)

# onnx 的 elem_type -> numpy dtype
ONNX2NP = {1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32, 7: np.int64,
           10: np.float16, 11: np.float64}


def _input_meta(path):
    m = onnx.load(path)
    out = []
    for i in m.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 1
                 for d in i.type.tensor_type.shape.dim]
        dtype = ONNX2NP.get(i.type.tensor_type.elem_type, np.float32)
        out.append((i.name, shape, dtype))
    return out


class RandomCalib(CalibrationDataReader):
    """用随机张量做校准（无真实数据时的兜底；有数据时请用 --calib-npz）。"""

    def __init__(self, inputs, n=8, seed=0, npz=None):
        self.inputs = inputs
        self.n = n
        self.it = 0
        self.npz = np.load(npz) if npz else None
        rng = np.random.default_rng(seed)
        self.batch = []
        for _ in range(n):
            feed = {}
            for name, shape, dtype in inputs:
                if self.npz is not None and name in self.npz:
                    arr = self.npz[name]
                    idx = min(self.it, arr.shape[0] - 1)
                    feed[name] = arr[idx:idx + 1].astype(dtype)
                else:
                    # 默认输入幅度：图像 0~255 → 归一化后 ±2；其余用标准正态
                    if dtype == np.float16:
                        dtype = np.float16
                    feed[name] = rng.standard_normal(shape).astype(dtype) * 0.5
            self.batch.append(feed)
            self.it += 1
        self.it = 0

    def get_next(self):
        if self.it >= len(self.batch):
            return None
        feed = self.batch[self.it]
        self.it += 1
        return feed


def quantize(path, out, n_calib=8, calib_npz=None, per_channel=True, dynamic=False):
    inputs = _input_meta(path)
    print(f"[int8] {os.path.basename(path)}  输入: "
          + ", ".join(f"{n}{s}" for n, s, _ in inputs))
    if dynamic:
        # 动态量化：不需要校准，权重转 int8、激活运行时量化（对 Conv 收益有限但稳定）
        quantize_dynamic(path, out, weight_type=QuantType.QInt8,
                         per_channel=per_channel)
    else:
        reader = RandomCalib(inputs, n=n_calib, npz=calib_npz)
        quantize_static(
            path, out, reader,
            quant_format=QuantFormat.QDQ,      # 部署端（TensorRT）认 QDQ
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
        )
    m = onnx.load(out)
    n_qdq = sum(1 for n in m.graph.node
                if n.op_type in ("QuantizeLinear", "DequantizeLinear"))
    print(f"[int8] -> {out}  ({os.path.getsize(out)/2**20:.1f} MB, QDQ 节点 {n_qdq})")
    # 快速跑一遍确认图能加载
    try:
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        feed = {i.name: np.random.randn(*[d for d in i.shape]).astype(np.float32)
                for i in sess.get_inputs()}
        sess.run(None, feed)
        print("[int8] onnxruntime 推理自检通过")
    except Exception as e:  # 自定义算子（如 spconv）会在这里失败
        print(f"[int8] 跳过自检: {type(e).__name__}: {str(e)[:80]}")


def main():
    ap = argparse.ArgumentParser(description="fp32 ONNX -> INT8 QDQ ONNX（onnxruntime PTQ）")
    ap.add_argument("onnx", nargs="+", help="输入 fp32 onnx（可多个）")
    ap.add_argument("--out", default=None, help="输出路径（单文件时有效，默认 <name>.int8.onnx）")
    ap.add_argument("--calib", type=int, default=8, help="校准样本数")
    ap.add_argument("--calib-npz", default=None, help="真实校准数据（键=输入名）")
    ap.add_argument("--no-per-channel", action="store_true")
    ap.add_argument("--dynamic", action="store_true", help="动态量化（免校准）")
    args = ap.parse_args()

    for src in args.onnx:
        out = args.out or (os.path.splitext(src)[0] + ".int8.onnx")
        try:
            quantize(src, out, n_calib=args.calib, calib_npz=args.calib_npz,
                     per_channel=not args.no_per_channel, dynamic=args.dynamic)
        except Exception as e:
            print(f"[int8][skip] {os.path.basename(src)}: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
