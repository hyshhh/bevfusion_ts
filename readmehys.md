# 部署导出（fp16）

# 环境
cd /home/lab-203/hys/BEV/bevfusion
conda activate bevfusion

# 一条命令导出（ckpt fp16|int8 [config] [num_cams] [per_class_topk]）
bash run_export.sh runs/yz_z8_160ft_pscore/best_mAP.pth fp16 \
     configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml 3

# 老模型（默认 config 即 camera+lidar_180x180_spconv.yaml）
bash run_export.sh runs/yz_spconv_z21/epoch_100.pth fp16

# 需要临时改按类配额时给第 5 个参数（不填则用 yaml 里的）
bash run_export.sh <ckpt> fp16 <config> 3 "20,28,32,20,48,24,36,20,36"

# 分步执行
# 训练 ckpt -> 整模型 pickle（config 路径已烘进 pickle）
python tools/to_deploy_ckpt.py \
    configs/YZdata/180x180_20_z8_pctopk_ada_pscore.yaml \
    runs/yz_z8_160ft_pscore/best_mAP.pth --num-cams 3

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

# 产出（qat/onnx_fp16/）
ls qat/onnx_fp16/
# camera.onnx        img[1,3,256,704]      -> feat[1,256,32,88]
# depthnet.onnx      feat[1,256,32,88]     -> depth_context
# fuser.onnx         cam_bev+lidar_bev     -> middle[1,512,180,180]
# head.bbox.onnx     middle[1,512,180,180] -> score/rot/dim/reg/height/vel
# lidar.backbone.onnx points               -> lidar_bev[1,128,180,180]

# 注意：不要加这个变量，会让 MAGMA 在 torch.inverse 崩（exit 134）
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 注意：权重和 yaml 必须配套，否则 load_checkpoint 会跳过不匹配的层
# 导出脚本启动会打印 [info] deploy cfg: <pickle 里记录的 yaml 路径>

# 无法 ONNX 导出的部分（部署端自己实现）
# LiDAR sparse conv  走部署框架的权重转换
# bev_pool           自定义 CUDA kernel（视锥 -> BEV）
# 检测头后处理       topk / circle-NMS
