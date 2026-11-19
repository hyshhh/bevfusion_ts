# 部署导出（fp16 / int8）

# 环境
cd /home/lab-203/hys/BEV/bevfusion
conda activate bevfusion

# ---------------------------------------------------------------- 校准数据（int8 用）
# 在真实 val 帧上跑前向，dump 各 ONNX 模块的输入张量（16 帧，npz 约 129MB）
python tools/dump_calib.py \
    configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml \
    runs/yz_z8_160ft_pscore/best_mAP.pth \
    --ann data/data_pkl/custom_3000/infos_val.pkl \
    --num 16 --out calib/yzz8_calib16.npz
# 产出键名 = ONNX 输入名：
#   img     (16, 1, 3, 256, 704)     相机预处理图
#   depth   (16, 1, 1, 256, 704)     全零占位
#   camera  (16, 64, 180, 180)       相机 BEV（fuser 输入）
#   lidar   (16, 128, 180, 180)      LiDAR BEV（fuser 输入）
#   middle  (16, 512, 180, 180)      head 输入（decoder neck 输出）

# ---------------------------------------------------------------- 一键导出
# fp16（本项目单相机，num_cams=1）
bash run_export.sh runs/yz_z8_160ft_pscore/best_mAP.pth fp16 \
     configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml 1

# int8（用上面的 npz 做校准）
CALIB_NPZ=calib/yzz8_calib16.npz bash run_export.sh \
    runs/yz_z8_160ft_pscore/best_mAP.pth int8 \
    configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml 1

# ---------------------------------------------------------------- fp16 分步
# 训练 ckpt -> 整模型 pickle（config 路径、per_class_topk 已烘进 pickle）
python tools/to_deploy_ckpt.py \
    configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml \
    runs/yz_z8_160ft_pscore/best_mAP.pth --num-cams 1

# 后续步骤只认这个 pickle
DEPLOY_CKPT=qat/ckpt/deploy_best_mAP.pth

# 相机主干 + vtransform
python qat/export-camera.py --ckpt=$DEPLOY_CKPT --fp16 --device cpu

# fuser + 检测头（复用 TransFusionHead.forward_single，含按类 top-K）
python qat/export-transfuser.py --ckpt=$DEPLOY_CKPT --fp16 --device cpu

# 雷达 SCN
python qat/export-scn.py --ckpt=$DEPLOY_CKPT \
    --save=qat/onnx_fp16/lidar.backbone.onnx --device cpu

# 打包
tar czf deploy_fp16.tgz -C qat/onnx_fp16 .

# ---------------------------------------------------------------- int8 分步
# fp32 ONNX（head 必须加 --std-layernorm，nv:: 自定义算子 ORT 不认）
python qat/export-camera.py --ckpt=$DEPLOY_CKPT --device cpu
python qat/export-transfuser.py --ckpt=$DEPLOY_CKPT --device cpu --std-layernorm
python qat/export-scn.py --ckpt=$DEPLOY_CKPT \
    --save=qat/onnx_int8/lidar.backbone.onnx --device cpu

# fp32 ONNX -> int8 QDQ（用真实校准数据）
python qat/quantize_onnx_int8.py \
    qat/onnx_int8/camera.backbone.onnx \
    qat/onnx_int8/head.bbox.onnx \
    --calib 16 --calib-npz calib/yzz8_calib16.npz

# 打包
tar czf deploy_int8.tgz -C qat/onnx_int8 .

# ---------------------------------------------------------------- int8 实测
# camera.backbone  98.1MB -> 25.0MB  相对 L2 13.9%  余弦 0.990  ✅
# head.bbox         5.3MB ->  2.6MB  相对 L2 18.1%  余弦 0.979  ✅（需端上验 mAP）
# fuser            19.1MB  保持 fp32  相对 L2 94~125%        ❌ ORT PTQ 量化崩，交给 TensorRT 校准
# lidar.backbone    5.1MB  保持 fp32  spconv 自定义算子 ORT 不支持

# ---------------------------------------------------------------- int8 QAT（量化感知训练）
# ① 修 pytorch_quantization：cuda_ext 与 torch 1.10.1 ABI 不匹配，打补丁换成纯 Torch fake-quant
python qat/patch_pytorch_quantization.py

# ② 快速路径：不训练，只校准 32 帧就把 quantizer 的 amax 定下来，直接出 QDQ 模型
python tools/qat_calibrate.py \
    configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml \
    runs/yz_z8_160ft_pscore/best_mAP.pth \
    --ann data/data_pkl/custom_3000/infos_val.pkl \
    --num 32 --out qat/ckpt/deploy_qat.pth

# ③ QAT 微调：Conv1d/2d/3d + Linear 全部插入 fake-quant（实测 282 个量化模块）
#    --calibrate 必给：不给的话 quantizer 停在 dynamic 模式（amax 不进 state_dict，ckpt 里没有量化参数）
#    histogram + 99.99 分位：稀疏 BEV 用 max 校准会被离群峰毁掉（实测 mAP 0.69 vs 0.90）
#    fp32 训练（避免 fp16 叠加溢出），lr 要小，只让权重适应量化误差
#    直接 python 启动即可（脚本内部会自建 world_size=1 进程组；也可以套 torchpack dist-run -np 1）
python tools/train_qat.py \
    configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml \
    --run-dir runs/yz_qat \
    --load-from runs/yz_z8_2002/epoch_200.pth \
    --calibrate 32 --calib-method histogram --calib-percentile 99.99 \
    max_epochs=10 data.samples_per_gpu=3 use_fp16=False optimizer.lr=1e-5
# 评测量化模型（不训练，现场校准后直接出 mAP）
python tools/eval_qat.py \
    configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml \
    runs/yz_qat/epoch_10.pth \
    --calibrate 32 --calib-method histogram --calib-percentile 99.99
# QAT 权重再校准一次，产出带 amax 的可导出 pickle
python tools/qat_calibrate.py \
    configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml \
    runs/yz_qat/epoch_10.pth \
    --ann data/data_pkl/custom_3000/infos_val.pkl \
    --num 32 --out qat/ckpt/deploy_qat.pth

# ④ 导出 QDQ ONNX（head 必须 --std-layernorm，否则 ORT 不认 nv:: 自定义算子）
python qat/export-camera.py --ckpt=qat/ckpt/deploy_qat.pth --device cpu
python qat/export-transfuser.py --ckpt=qat/ckpt/deploy_qat.pth --device cpu --std-layernorm
python qat/export-scn.py --ckpt=qat/ckpt/deploy_qat.pth \
    --save=qat/onnx_int8/lidar.backbone.onnx --device cpu

# ⑤ 校验 Q/DQ 节点（实测：camera.backbone 112 对、head.bbox 44 对）
python - <<'PY'
import onnx, collections
for p in ["qat/onnx_int8/camera.backbone.onnx", "qat/onnx_int8/head.bbox.onnx"]:
    m = onnx.load(p); ops = collections.Counter(n.op_type for n in m.graph.node)
    print(p, "QDQ =", ops.get("QuantizeLinear", 0))
PY
