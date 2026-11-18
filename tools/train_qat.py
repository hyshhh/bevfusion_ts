"""QAT（量化感知训练）入口：在 build_model 之前把 Conv2d/Linear 换成带 fake-quant 的版本。

与 tools/train.py 的唯一区别：
    quant_modules.initialize()  →  nn.Conv2d/nn.Linear 被替换为 QuantConv2d/QuantLinear，
                                  前向里插入 int8 伪量化（可学习/可校准的 amax）
其余（config、数据集、runner、checkpoint、评测 hook）完全复用。

用法（与 train.py 相同，只是换成这个脚本）:
    python tools/train_qat.py <config> --run-dir runs/xxx \
        --load-from runs/yz_z8_160ft_pscore/best_mAP.pth \
        max_epochs=10 data.samples_per_gpu=4 use_fp16=False optimizer.lr=1e-5

注意：
  - QAT 建议 fp32 训练（量化本身已经降低数值范围，fp16 叠加容易溢）
  - 训练完的 ckpt 里带着 quantizer 的 amax，可直接进 qat/export-*.py 出 QDQ ONNX
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def enable_qat():
    """打补丁：Conv2d/Linear → 带 fake-quant 的版本，并启用伪量化。"""
    try:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization import quant_modules
    except ImportError as e:
        raise RuntimeError(
            "pytorch_quantization 不可用（cuda_ext ABI 不匹配 → 已改为纯 PyTorch 回退，"
            f"若仍失败请重装）：{e}")
    # 用 torch 的 fake-quant 算子（不依赖 cuda_ext）
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    # 注意：initialize() 会把 torch.nn.Conv2d 直接换成 QuantConv2d，
    # 所以必须在它之前记住"哪些注册项原本是 torch.nn 的类"，用类名匹配，
    # 不能靠对象 identity 判断。
    import torch.nn as nn
    from mmcv.cnn import CONV_LAYERS

    def _orig_names():
        return {nn.Conv1d.__name__: "QuantConv1d",
                nn.Conv2d.__name__: "QuantConv2d",
                nn.Conv3d.__name__: "QuantConv3d",
                nn.Linear.__name__: "QuantLinear"}

    name_map = _orig_names()
    quant_modules.initialize()

    # mmcv 的 build_conv_layer 走 CONV_LAYERS 注册表，注册表里存的是导入时绑定的
    # 原始类；不换掉它，ResNet / TransFusion 里用 build_conv_layer 搭的层就不会被量化。
    n_swapped = 0
    for key, val in list(CONV_LAYERS._module_dict.items()):
        qn = name_map.get(getattr(val, "__name__", ""))
        qcls = getattr(quant_nn, qn, None) if qn else None
        if qcls is not None and not issubclass(val, qcls):
            CONV_LAYERS._module_dict[key] = qcls
            n_swapped += 1
    print(f"[QAT] mmcv CONV_LAYERS 替换 {n_swapped} 项")
    return quant_nn


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument("--resume-from", metavar="FILE")
    parser.add_argument("--load-from", metavar="FILE",
                        help="QAT 初始化权重（通常是 PTQ/FP32 训练出来的 ckpt）")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.load_from:
        cfg.load_from = args.load_from

    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logger = get_root_logger(log_file=os.path.join(cfg.run_dir, f"{timestamp}.log"))
    logger.info(f"[QAT] 启用量化感知：Conv2d/Linear → QuantConv2d/QuantLinear")

    # 关键：必须在 build_model 之前 patch，否则模型里还是普通 Conv
    enable_qat()

    if cfg.seed is not None:
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]
    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    n_q = sum(1 for m in model.modules()
              if type(m).__name__.startswith("Quant") or "quantizer" in type(m).__name__.lower())
    logger.info(f"[QAT] 已量化模块数: {n_q}")
    train_model(model, datasets, cfg, distributed=True, validate=True, timestamp=timestamp)


if __name__ == "__main__":
    main()
