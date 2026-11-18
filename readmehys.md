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

# int8 实测（新权重，单相机）
# camera.backbone  98.1MB -> 25.0MB  Q/DQ 82/198
# fuser            19.1MB ->  4.8MB  Q/DQ 20/51
# head.bbox         5.3MB ->  2.6MB  Q/DQ 123/180
# lidar.backbone    跳过：spconv 自定义算子 onnxruntime 不支持，走部署框架自己的转换

# 产出（qat/onnx_fp16/ 或 qat/onnx_int8/）
ls qat/onnx_fp16/
# camera.backbone.onnx  img[1,1,3,256,704]   -> feat / depth_weights
# camera.vtransform.onnx feat_in             -> bev_pool 用
# fuser.onnx            cam_bev+lidar_bev    -> middle[1,512,180,180]
# head.bbox.onnx        middle[1,512,180,180]-> score/rot/dim/reg/height/vel
# lidar.backbone.onnx   points               -> lidar_bev[1,128,180,180]

# 部署端精度开关（TensorRT）
# fp16：trtexec --onnx=xxx.onnx --fp16 --saveEngine=xxx.engine
# int8：ONNX 已带 QDQ，直接 --int8 构建；若用 fp32 图则需 --int8 --calib=cache.txt
# 混精：config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS) 后逐层设 precision

# 注意：不要加这个变量，会让 MAGMA 在 torch.inverse 崩（exit 134）
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 注意：权重和 yaml 必须配套，否则 load_checkpoint 会跳过不匹配的层
# 导出脚本启动会打印 [info] deploy cfg: <pickle 里记录的 yaml 路径>

# 无法 ONNX 导出的部分（部署端自己实现）
# LiDAR sparse conv  走部署框架的权重转换
# bev_pool           自定义 CUDA kernel（视锥 -> BEV）
# 检测头后处理       topk / circle-NMS
