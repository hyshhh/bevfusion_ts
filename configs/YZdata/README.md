# YZdata 配置（custom_3000 水面目标）

单相机 + 单 LiDAR + ResNet-50 相机分支，LiDAR 稀疏卷积分支用 TorchSparse++。

## 环境

```bash
conda activate bevfusion_ts
cd /home/lab-203/hys/BEV/bevfusion_ts
```
we
## 训练
```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/YZdata/180_180_20.yaml \
    --run-dir runs/ts_fp16_b16 \
    max_epochs=200 \
    data.samples_per_gpu=8 \
    use_fp16=True
    
torchpack dist-run -np 1 python tools/test.py \
    configs/YZdata/180_180_20.yaml \
    runs/ts_fp16_b16/epoch_200.pth --out result.pkl

python demo/vis_pred.py \
    configs/YZdata/180_180_20.yaml \
    result.pkl --score 0.3 --limit 1000 --out-dir viz/pred

python demo/eval_bev_ap.py \
    configs/YZdata/180_180_20.yaml result.pkl --iou 0.3 --ranges 0-30 30-50 50-80
```
纯 fp32：
```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/YZdata/camera+lidar_resnet50.yaml \
    --run-dir runs/ts_fp32_b16 \
    max_epochs=200 \
    data.samples_per_gpu=16
```

断点续训：

```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/YZdata/camera+lidar_resnet50.yaml \
    --run-dir runs/ts_fp16_b16 \
    --resume-from runs/ts_fp16_b16/latest.pth
```

## 基准测试
```bash
torchpack dist-run -np 1 python configs/YZdata/bench.py \
    configs/YZdata/camera+lidar_resnet50.yaml \
    --batches 8 10 16 20 24

torchpack dist-run -np 1 python configs/YZdata/bench.py \
    configs/YZdata/camera+lidar_resnet50.yaml \
    --fp16 --batches 8 10 16 20 24
```
## 测试

```bash
torchpack dist-run -np 1 python tools/test.py \
    configs/YZdata/camera+lidar_resnet50.yaml \
    runs/ts_fp16_b16/latest.pth \
    --out result.pkl
```

## 可视化

```bash
python demo/vis_gt.py \
    configs/YZdata/camera+lidar_resnet50.yaml --limit 20 --out-dir viz/gt

python demo/vis_gt.py \
    configs/YZdata/camera+lidar_resnet50.yaml --limit 20 --out-dir viz/gt --zoom 1

python demo/vis_gt.py \
    configs/YZdata/camera+lidar_resnet50.yaml --limit 20 --out-dir viz/gt --y-range 40
```

## 导出 ONNX
```bash
python tools/export.py \
    configs/YZdata/camera+lidar_resnet50.yaml \
    runs/ts_fp16_b16/latest.pth \
    --out bev_camera.onnx
```
