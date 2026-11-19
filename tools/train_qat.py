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


def enable_qat(calib_method="max", per_channel_weight=True):
    """打补丁：Conv2d/Linear → 带 fake-quant 的版本，并启用伪量化。

    Args:
        calib_method: "max" 或 "histogram"（稀疏 BEV 建议 histogram，避开离群峰）
        per_channel_weight: 权重用 per-channel 量化（更准，部署端 TRT 也支持）
    """
    try:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization import quant_modules
        from pytorch_quantization.tensor_quant import QuantDescriptor
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

    # 校准方式必须在 quant_modules.initialize() 之前设到各类的默认描述符上，
    # 否则 quantizer 构造完就固定成 max 校准了。
    desc_in = QuantDescriptor(num_bits=8, calib_method=calib_method)
    desc_w = QuantDescriptor(num_bits=8, axis=(0,)) if per_channel_weight \
        else QuantDescriptor(num_bits=8)
    for _n in ("QuantConv1d", "QuantConv2d", "QuantConv3d", "QuantLinear"):
        _cls = getattr(quant_nn, _n, None)
        if _cls is None:
            continue
        _cls.default_quant_desc_input = desc_in
        _cls.default_quant_desc_weight = desc_w

    name_map = _orig_names()
    quant_modules.initialize()
    print(f"[QAT] 校准方式={calib_method} | 权重 per-channel={per_channel_weight}")

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


def calibrate_static(model, cfg, n, device="cuda", percentile=99.99):
    """用 n 帧真实数据统计 amax，并把 quantizer 切成"静态 fake-quant"。

    关键点：TensorQuantizer 默认不注册 _amax buffer（QuantDescriptor.amax=None），
    此时处于 dynamic 模式 —— 每次 forward 现算 scale，既不可保存也偏离部署形态。
    只有调用 load_calib_amax()（内部走 amax setter → register_buffer）之后，
    _amax 才进 state_dict，ckpt 才会带着量化参数。
    """
    import torch
    from mmcv.parallel import MMDataParallel, collate
    from mmdet3d.datasets import build_dataset

    qs = [m for m in model.modules() if "TensorQuantizer" in type(m).__name__]
    if not qs:
        raise RuntimeError("没有找到 TensorQuantizer，QAT 未生效")
    for q in qs:
        q.enable_calib()
    print(f"[QAT] 校准 {n} 帧，quantizer {len(qs)} 个")

    ann = cfg.get("ann_val", None)
    ds = build_dataset(dict(cfg.data.test, ann_file=ann) if ann else cfg.data.test)
    wrapper = MMDataParallel(model.to(device), device_ids=[0]).eval()
    ok = 0
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            data = collate([ds[i]], samples_per_gpu=1)
            try:
                wrapper(return_loss=False, **data)
                ok += 1
            except Exception as e:
                print(f"  [skip] 帧 {i}: {type(e).__name__}: {str(e)[:60]}")
    if ok == 0:
        raise RuntimeError("校准一帧都没成功")

    for q in qs:
        # MaxCalibrator 无参；HistogramCalibrator 必须给 method（percentile 避开离群峰）
        try:
            q.load_calib_amax()
        except TypeError:
            q.load_calib_amax(method="percentile", percentile=percentile)
        q.disable_calib()
        q.use_fb_fake_quant = True
    # histogram 校准的 amax 落在 CPU 上，搬回 GPU 再继续训练
    model.to(device)
    model.train()
    n_has = sum(1 for q in qs if hasattr(q, "_amax"))
    print(f"[QAT] 校准完成：{ok}/{n} 帧成功，{n_has}/{len(qs)} 个 quantizer 已有 amax")
    return model


def main():
    # 兼容两种启动方式：
    #   ① torchpack dist-run -np 1 python tools/train_qat.py ...
    #   ② 直接 python tools/train_qat.py ...（没有 MASTER_HOST，走单卡模式）
    is_dist = "MASTER_HOST" in os.environ
    if is_dist:
        dist.init()
        local_rank = dist.local_rank()
        print(f"[QAT] 分布式模式 local_rank={local_rank}")
    else:
        # mmdet3d 的 train_model 里 build_dataloader 传的是 num_gpus=None，
        # 只有 dist=True 时才会用 get_dist_info() 补上 world_size；
        # 所以直接 python 启动时也要手工建一个 world_size=1 的进程组。
        import torch.distributed as tdist
        os.environ.setdefault("LOCAL_RANK", "0")
        local_rank = 0
        if not tdist.is_initialized():
            tdist.init_process_group(backend="nccl",
                                     init_method="tcp://127.0.0.1:29511",
                                     world_size=1, rank=0)
        print("[QAT] 单卡模式（未用 torchpack 启动，已建 world_size=1 进程组）")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument("--resume-from", metavar="FILE")
    parser.add_argument("--load-from", metavar="FILE",
                        help="QAT 初始化权重（通常是 PTQ/FP32 训练出来的 ckpt）")
    parser.add_argument("--calibrate", type=int, default=0,
                        help="训练前用 N 帧校准 amax（0=不校准，会退化成 dynamic 量化）")
    parser.add_argument("--calib-method", default="histogram", choices=["max", "histogram"],
                        help="校准方式，默认 histogram（稀疏 BEV 用 max 会被离群值毁掉）")
    parser.add_argument("--calib-percentile", type=float, default=99.99,
                        help="histogram 校准的百分位（越小越激进地裁掉离群值）")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(local_rank)

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
    enable_qat(calib_method=args.calib_method)

    if cfg.seed is not None:
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]
    model = build_model(cfg.model)
    model.init_weights()
    # 先用 --load-from 权重做校准（train_model 内部还会再 load 一次，幂等）
    if args.calibrate > 0:
        if args.load_from:
            from mmcv.runner import load_checkpoint
            load_checkpoint(model, args.load_from, map_location="cpu")
        model = calibrate_static(model, cfg, args.calibrate,
                                 percentile=args.calib_percentile)
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
