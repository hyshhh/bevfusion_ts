"""校验导出的 ONNX 结构是否正确：形状 + 与 PyTorch 的数值一致性。

做法：在真实帧上同时跑 PyTorch 子模块和对应 ONNX，逐输出比较。
    camera.backbone : img -> (camera_feature, camera_depth_weights)
    fuser           : (camera, lidar) -> middle
    head.bbox       : middle -> (score, rot, dim, reg, height, vel)

用法:
    python qat/verify_onnx.py <config> <ckpt> [--onnx-dir qat/onnx_int8] \
        [--qat] [--num 4] [--skip camera,fuser,head]
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs as tp_configs

import mmdet3d  # noqa: F401
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def ort_run(path, feeds):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feed = {k: v for k, v in feeds.items() if k in names}
    outs = sess.run(None, feed)
    return names, [i.name for i in sess.get_outputs()], outs


def ort_run_batched(path, feeds_per_frame):
    """ONNX 按 batch=1 导出，这里逐帧跑再把结果拼起来。"""
    names = None
    acc = None
    for f in feeds_per_frame:
        names, onames, outs = ort_run(path, f)
        if acc is None:
            acc = [[] for _ in outs]
        for k, o in enumerate(outs):
            acc[k].append(o)
    return onames, [np.concatenate(x, axis=0) for x in acc]


def compare(tag, a_list, b_list, a_names, b_names):
    print(f"\n--- {tag}")
    for a, b, an, bn in zip(a_list, b_list, a_names, b_names):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            print(f"  {bn:24} 形状不一致: pytorch{a.shape} vs onnx{b.shape}")
            continue
        denom = np.linalg.norm(a) + 1e-12
        rel = np.linalg.norm(a - b) / denom
        cos = float((a.ravel() @ b.ravel()) / (denom * (np.linalg.norm(b) + 1e-12)))
        print(f"  {bn:24} shape={tuple(b.shape)}  相对L2={rel*100:6.3f}%  cos={cos:.6f}  "
              f"max|Δ|={np.abs(a-b).max():.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--onnx-dir", default="qat/onnx_int8")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--qat", action="store_true", help="按 QAT 方式建模型（ckpt 里有 amax）")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()

    tp_configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(tp_configs), filename=args.config)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    if args.qat:
        from tools.train_qat import enable_qat
        enable_qat()

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

    ann = args.ann or cfg.ann_val
    ds = build_dataset(dict(cfg.data.test, ann_file=ann))

    grabbed = {}
    h_backbone = model.module.encoders.camera.backbone.register_forward_hook(
        lambda m, i, o: grabbed.setdefault("backbone_out", []).append(o))
    h_fuser_pre = model.module.fuser.register_forward_pre_hook(
        lambda m, i: grabbed.setdefault("fuser_in", []).append(i[0]))
    h_fuser = model.module.fuser.register_forward_hook(
        lambda m, i, o: grabbed.setdefault("fuser_out", []).append(o))
    h_head_pre = model.module.heads["object"].register_forward_pre_hook(
        lambda m, i: grabbed.setdefault("head_in", []).append(i[0]))

    # 直接复用导出脚本里的包装类，保证"PyTorch 侧"与导出的语义完全一致
    import importlib.util
    spec = importlib.util.spec_from_file_location("export_camera", "qat/export-camera.py")
    ecam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ecam)
    cam_wrapper = ecam.SubclassCameraModule(model.module).cuda().eval()

    cam_pt, fuse_pt, head_pt = {}, {}, {}
    with torch.no_grad():
        for i in range(args.num):
            data = collate([ds[i]], samples_per_gpu=1)
            img = data["img"].data[0].cuda()
            if img.dim() == 4:
                img = img.unsqueeze(0)
            dummy_depth = torch.zeros(1, 1, 1, img.shape[-2], img.shape[-1], device="cuda")
            cf, cd = cam_wrapper(img, dummy_depth)
            cam_pt.setdefault("feature", []).append(cf.cpu().numpy())
            cam_pt.setdefault("depth", []).append(cd.cpu().numpy())
            cam_pt.setdefault("img", []).append(img.cpu().numpy())
            model(return_loss=False, **data)   # 触发 fuser / head 的 hook
    for h in (h_backbone, h_fuser_pre, h_fuser, h_head_pre):
        h.remove()

    feats = grabbed.get("fuser_in", [])
    if feats and "fuser" not in skip:
        cam_bev = torch.cat([f[0] for f in feats]).cpu().numpy()
        lid_bev = torch.cat([f[1] for f in feats]).cpu().numpy()
        # 导出的是 SubclassFuser（fuser + decoder.backbone + decoder.neck），
        # 所以 PyTorch 参考也要走同一个包装，而不是只看 model.fuser
        import importlib.util
        spec2 = importlib.util.spec_from_file_location("export_tf", "qat/export-transfuser.py")
        etf = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(etf)
        fuser_wrapper = etf.SubclassFuser(model.module).cuda().eval()
        with torch.no_grad():
            middle_pt = torch.cat([
                fuser_wrapper([f[0].cuda(), f[1].cuda()]) for f in feats]).cpu().numpy()
        p = os.path.join(args.onnx_dir, "fuser.onnx")
        feeds = [{"camera": cam_bev[i:i+1], "lidar": lid_bev[i:i+1]}
                 for i in range(cam_bev.shape[0])]
        onames, outs = ort_run_batched(p, feeds)
        compare("fuser", [middle_pt], outs, ["middle(pytorch)"], onames)

    mids = grabbed.get("head_in", [])
    if mids and "head" not in skip:
        mid = torch.cat([m if torch.is_tensor(m) else m[0] for m in mids]).cpu().numpy()
        head = model.module.heads["object"]
        scores, rots, dims, centers, heights = [], [], [], [], []
        with torch.no_grad():
            for i in range(mid.shape[0]):        # 逐帧（ONNX 是 batch=1 导出的）
                pred = head.forward_single(torch.from_numpy(mid[i:i+1]).cuda(), None, None)[0]
                oh = torch.eye(head.num_classes, device="cuda")[head.query_labels.reshape(-1)]
                oh = oh[None].permute(0, 2, 1)
                scores.append((pred["heatmap"].sigmoid() * pred["query_heatmap_score"] * oh).cpu().numpy())
                rots.append(pred["rot"].cpu().numpy()); dims.append(pred["dim"].cpu().numpy())
                centers.append(pred["center"].cpu().numpy()); heights.append(pred["height"].cpu().numpy())
        pt = [np.concatenate(scores), np.concatenate(rots), np.concatenate(dims),
              np.concatenate(centers), np.concatenate(heights)]
        p = os.path.join(args.onnx_dir, "head.bbox.onnx")
        feeds = [{"middle": mid[i:i+1]} for i in range(mid.shape[0])]
        onames, outs = ort_run_batched(p, feeds)
        # onnx 输出: score, rot, dim, reg, height, vel
        compare("head.bbox", pt, [outs[0], outs[1], outs[2], outs[3], outs[4]],
                ["score", "rot", "dim", "center", "height"], onames[:5])

    if "camera" not in skip and cam_pt:
        p = os.path.join(args.onnx_dir, "camera.backbone.onnx")
        imgs = np.concatenate(cam_pt["img"]).astype(np.float32)
        feeds = [{"img": imgs[i:i+1],
                  "depth": np.zeros((1, 1, 1, imgs.shape[-2], imgs.shape[-1]), np.float32)}
                 for i in range(imgs.shape[0])]
        onames, outs = ort_run_batched(p, feeds)
        compare("camera.backbone",
                [np.concatenate(cam_pt["feature"]), np.concatenate(cam_pt["depth"])],
                outs, ["camera_feature", "camera_depth_weights"], onames)


if __name__ == "__main__":
    main()
