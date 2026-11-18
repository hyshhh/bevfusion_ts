# 部署导出（fp16 / int8）

# 环境
cd /home/lab-203/hys/BEV/bevfusion
conda activate bevfusion

# 一条命令导出（ckpt fp16|int8 [config] [num_cams] [per_class_topk]）
# 本项目是单相机，num_cams 写 1
bash run_export.sh runs/yz_z8_160ft_pscore/best_mAP.pth fp16 \
     configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml 1
# 老模型（默认 config 即 camera+lidar_180x180_spconv.yaml）
bash run_export.sh runs/yz_spconv_z21/epoch_100.pth fp16

# 需要临时改按类配额时给第 5 个参数（不填则用 yaml 里的）
bash run_export.sh <ckpt> fp16 <config> 1 "20,28,32,20,48,24,36,20,36"

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

# ---------------------------------------------------------------- int8 导出
# int8 一条命令：先导 fp32 ONNX，再 onnxruntime 静态 PTQ 出 QDQ，最后打包
# 不需要 pytorch_quantization（本环境它 ABI 不匹配）
bash run_export.sh runs/yz_z8_160ft_pscore/best_mAP.pth int8 \
     configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml 1

# 校准样本数（默认 8 个随机张量）；有真实校准数据时给 npz（键名 = ONNX 输入名）
CALIB_N=32 bash run_export.sh <ckpt> int8 <config> 1
CALIB_NPZ=calib.npz bash run_export.sh <ckpt> int8 <config> 1

# int8 分步：fp32 ONNX（注意 head 要加 --std-layernorm，nv:: 自定义算子 ORT 不认）
python qat/export-camera.py --ckpt=$DEPLOY_CKPT --device cpu
python qat/export-transfuser.py --ckpt=$DEPLOY_CKPT --device cpu --std-layernorm
python qat/export-scn.py --ckpt=$DEPLOY_CKPT \
    --save=qat/onnx_int8/lidar.backbone.onnx --device cpu
# fp32 ONNX -> int8 QDQ（可批量；spconv 模块会自动跳过）
python qat/quantize_onnx_int8.py qat/onnx_int8/*.onnx --calib 16
tar czf deploy_int8.tgz -C qat/onnx_int8 .