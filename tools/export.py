"""导出 ONNX。

BEVFusion 的融合部分无法导出：LiDAR 分支用 spconv（无对应 ONNX 算子），
相机 BEV 投影里有自定义 CUDA 算子 bev_pool，以及 torch.inverse（opset 不支持）。
所以这里导出可部署的部分——相机 encoder（backbone + neck）。

用法:
    python tools/export.py configs/YZdata/camera+lidar_resnet50.yaml [checkpoint] --out bev_camera.onnx
"""
import argparse

import onnx
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs


class CameraEncoder(torch.nn.Module):
    """只含相机 backbone + neck，输出融合后的特征图。"""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.encoders["camera"]["backbone"]
        self.neck = model.encoders["camera"]["neck"]

    def forward(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x if isinstance(x, torch.Tensor) else x[0]


def parse_args():
    parser = argparse.ArgumentParser(description="export camera encoder to ONNX")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", nargs="?", default=None, help="checkpoint file")
    parser.add_argument("--out", default="bev_camera.onnx", help="output onnx path")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    if not args.checkpoint:
        # 不加载权重时跳过 torchvision 预训练下载，导出的是结构
        cfg.model.encoders.camera.backbone.init_cfg = None
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()

    net = CameraEncoder(model).cuda()
    h, w = cfg.image_size
    dummy = torch.randn(1, 3, h, w, device="cuda")

    with torch.no_grad():
        torch.onnx.export(
            net,
            (dummy,),
            args.out,
            input_names=["img"],
            output_names=["feat"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes={"img": {0: "batch"}} if args.dynamic_batch else None,
        )

    onnx_model = onnx.load(args.out)
    if not args.no_simplify:
        try:
            from onnxsim import simplify

            simplified, ok = simplify(onnx_model)
            if ok:
                onnx_model = simplified
                onnx.save(onnx_model, args.out)
            else:
                print("onnxsim 简化失败，保留原始图")
        except Exception as e:  # noqa: BLE001
            print(f"onnxsim 跳过: {type(e).__name__}: {e}")
    onnx.checker.check_model(onnx_model)

    inp = onnx_model.graph.input[0]
    out = onnx_model.graph.output[0]
    print(f"已导出 {args.out}")
    print(
        "  input : {} {}".format(
            inp.name, [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
        )
    )
    print(
        "  output: {} {}".format(
            out.name, [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
        )
    )
    print(f"  节点数: {len(onnx_model.graph.node)}")
    print("说明: LiDAR 分支(spconv) 与 bev_pool 无法导出，故只导相机 encoder")


if __name__ == "__main__":
    main()
