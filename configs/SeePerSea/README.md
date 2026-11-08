## 环境

```bash
conda activate bevfusion_ts
cd /home/lab-203/hys/BEV/bevfusion_ts
```
## 训练
```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/SeePerSea/camera+lidar_resnet50.yaml \
    --run-dir runs/seepersea \
    max_epochs=60 \
    data.samples_per_gpu=16 \
    use_fp16=True
```
## 测试
```bash
torchpack dist-run -np 1 python tools/test.py \
    configs/SeePerSea/camera+lidar_resnet50.yaml \
    runs/seepersea/epoch_96.pth --out result_ss.pkl
```
## 可视化
```bash
python demo/vis_gt.py \
    configs/SeePerSea/camera+lidar_resnet50.yaml --limit 20 --out-dir viz/gt_ss
python demo/vis_pred.py \
    configs/SeePerSea/camera+lidar_resnet50.yaml \
    result_ss.pkl --score 0.3 --limit 20 --out-dir viz/pred_ss
```
## 评测
```bash
python demo/eval_bev_ap.py \
    configs/SeePerSea/camera+lidar_resnet50.yaml \
    result_ss.pkl --iou 0.5 --ranges 0-30 30-60 60-80
```
