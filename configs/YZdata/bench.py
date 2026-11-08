"""后端 / 精度对比基准：显存 + 速度，自动扫 batch 直到 OOM。

torchpack dist-run -np 1 python configs/YZdata/bench.py \
    configs/YZdata/camera+lidar_resnet50.yaml --batches 8 10 12 16 20 24
加 --fp16 测混合精度。
"""
import argparse, gc, time
import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import wrap_fp16_model
from mmdet3d.utils import recursive_eval
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from torchpack.utils.config import configs

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--batches", type=int, nargs="+", default=[8, 10, 12, 16, 20])
ap.add_argument("--iters", type=int, default=8)
ap.add_argument("--fp16", action="store_true")
args = ap.parse_args()

configs.load(args.config, recursive=True)
cfg = Config(recursive_eval(configs), filename=args.config)
cfg.model.encoders.camera.backbone.init_cfg = None
model = build_model(cfg.model).cuda().train()
if args.fp16:
    wrap_fp16_model(model)
    print("fp16: 已启用 autocast 混合精度", flush=True)
print("参数量 %.1f M" % (sum(x.numel() for x in model.parameters()) / 1e6), flush=True)

ds = build_dataset(cfg.data.train)

def settle():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

settle()
torch.zeros(1, device="cuda")
base = torch.cuda.mem_get_info()[0]

for bs in args.batches:
    try:
        dl = build_dataloader(ds, bs, 0, dist=False, shuffle=True)
        it = iter(dl)
        model.zero_grad(set_to_none=True)

        def step(b):
            with torch.cuda.amp.autocast(enabled=args.fp16):
                losses = model(**b)
                loss = sum(v for k, v in losses.items()
                           if "loss" in k and torch.is_tensor(v) and v.requires_grad)
            loss.backward()
            model.zero_grad(set_to_none=True)
            return float(loss.detach())

        b = scatter(next(it), [0])[0]               # 取数据不计时
        v = step(b)                                 # 首次含一次性显存分配
        torch.cuda.synchronize()
        used = (base - torch.cuda.mem_get_info()[0]) / 1e9
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            step(b)                                 # 复用同一个 batch，只测模型
        torch.cuda.synchronize()
        dt = (time.time() - t0) / args.iters
        print(f"batch={bs:3d}: {dt*1000:6.0f} ms/iter | {bs/dt:5.1f} 样本/s | "
              f"显存 {used:5.2f} GB | loss {v:.3f}", flush=True)
        del dl, it
        settle()
    except RuntimeError as e:
        msg = str(e).split("\n")[0][:70]
        print(f"batch={bs:3d}: 失败 -> {msg}", flush=True)
        model.zero_grad(set_to_none=True)
        settle()
