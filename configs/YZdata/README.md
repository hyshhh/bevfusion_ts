# YZdata 配置（custom_3000 水面目标）

```bash
conda activate bevfusion
cd /home/lab-203/hys/BEV/bevfusion
```
## 训练

**从零训练**（9 类，z∈[-12,8]，相机深度 80 m）：
```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml \
    --run-dir runs/yz_z8_200 \
    max_epochs=200 data.samples_per_gpu=6 use_fp16=True optimizer.lr=1.5e-4
```

**从已有权重微调**（`180x180_20_z8_update.yaml` = z8 配置，仅相机深度扩到 159 m / 1 m 粒度）：
```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/YZdata/180x180_20_z8_update.yaml \
    --run-dir runs/yz_z8_160ft \
    --load-from runs/yz_z8_200/epoch_137.pth \
    max_epochs=100 data.samples_per_gpu=6 use_fp16=True optimizer.lr=5e-5
```

**续训**（同一 run-dir，从最近 checkpoint 接上，会恢复 epoch/iter/优化器/scaler）：
```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/YZdata/180x180_20_z8_update.yaml \
    --run-dir runs/yz_z8_160ft \
    --resume-from runs/yz_z8_160ft/epoch_36.pth \
    max_epochs=100 data.samples_per_gpu=6 use_fp16=True optimizer.lr=5e-5
```

## 测试
```bash
torchpack dist-run -np 1 python tools/test.py \
    configs/YZdata/camera+lidar_180x180_spconv.yaml \
    runs/yz_spconv_z21/epoch_100.pth \
    --out result.pkl
```
## 可视化
GT：
```bash
python demo/vis_gt.py configs/YZdata/camera+lidar_180x180_spconv.yaml \
    --out-dir viz/gt --limit 20
```
```bash
python demo/vis_pred.py configs/YZdata/camera+lidar_180x180_spconv.yaml \
    result.pkl --out-dir viz/pred --limit 1000 --score 0.3
```
## 评测（BEV AP）
```bash
python demo/eval_bev_ap.py configs/YZdata/camera+lidar_180x180_spconv.yaml \
    result.pkl --iou 0.3 --ranges 0-30 30-50 50-80 80+ 0-80
```

## 导出 ONNX

一次导出全部可部署模块：

```bash
bash run_export.sh runs/yz_spconv_z21/epoch_100.pth fp16
python tools/export_modules.py \
    configs/YZdata/camera+lidar_180x180_spconv.yaml \
    runs/yz_spconv_z21/epoch_100.pth output/
```

```bash
python tools/export_modules.py <cfg> <ckpt> output/ --modules depthnet
```