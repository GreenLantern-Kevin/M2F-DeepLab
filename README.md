# M2F-DeepLab

RGB 与三通道地形影像的双流滑坡语义分割模型，基于 DeepLabV3+。

## 模型

RGB 和 DEM 使用独立编码器。有 DEM 时，真实地形特征进入融合模块；缺失 DEM 时，FHM 从 RGB 高层特征生成替代地形特征。MEF 通过空间门控、通道门控和残差连接融合两路特征，BEM 在解码器中生成边界特征并参与最终分割。

`has_dem` 显式标识每个样本是否有地形数据。训练时可随机隐藏部分已配对 DEM，让 FHM 获得分割监督；配对样本同时提供特征一致性监督。FHM 对 RGB 输入特征和 DEM 教师目标使用梯度截断。真实 DEM 在完整模态路径中直接进入融合模块。

| 实验 | `--model` | 配置 |
| --- | --- | --- |
| A | `single_deeplabv3plus_xception16` | RGB DeepLabV3+ |
| B | `m2f_deeplabv3plus_xception16_b` | 双编码器 + 高层特征拼接与 1×1 投影 |
| C | `m2f_deeplabv3plus_xception16_c` | B + FHM |
| D | `m2f_deeplabv3plus_xception16_d` | 双编码器 + FHM + MEF |
| E | `m2f_deeplabv3plus_xception16` | D + BEM，默认模型 |

A–E 使用相同的 Xception16 主干规格、ASPP 膨胀率和输入设置。B 的缺失模态使用零地形特征。B 的拼接发生在编码后的特征层，区别于原始 RGB 与 DEM 的六通道输入拼接。

## 数据与依赖

Python 3.10+。先按训练服务器的 CUDA 安装匹配的 PyTorch / torchvision，再安装依赖：

```bash
pip install -r requirements.txt
```

```text
BijieLandslide/VOC2012/
├── JPEGImages/             # RGB 影像
├── DEMImages/              # 对齐的三通道 uint8 地形图，可部分缺失
├── SegmentationClass/      # 0=背景，1=滑坡，255=忽略
├── EdgeClass/              # 0=非边界，255=边界（读取后转为 0/1）
└── ImageSets/Segmentation/
    ├── train.txt           # 每行一个样本文件名或主文件名
    └── val.txt
```

RGB、DEM、标签和边界图须空间对齐且尺寸一致。按文件主名匹配，支持 PNG、JPG、TIF 等格式。DEM 三通道顺序为高程、坡度表达、坡向表达，训练和推理使用相同编码与 ImageNet 均值/标准差。`tools/prepare_dem.py` 可从已配准的单通道 DEM 生成逐图归一化的高程/Sobel 梯度幅度/方向编码；它不执行地理配准，梯度通道也不是经过像元尺度校准的物理坡度。

生成边界标签（所有模型共用同一数据接口）：

```bash
python tools/Generate_Edge_From_Mask.py --mask_dir BijieLandslide/VOC2012/SegmentationClass --output_edge_dir BijieLandslide/VOC2012/EdgeClass
```

若分割标签用 0/255 表示背景/滑坡，生成边界和训练时都加 `--mask_encoding binary255`。

## 训练与验证

以下命令从随机初始化开始：

```bash
python train.py --data_root BijieLandslide --no-pretrained_backbone --eval_missing_dem
```

若使用 Xception ImageNet 初始化，将对应权重放在 `weight/xception_pytorch_imagenet.pth`，并去掉 `--no-pretrained_backbone`。数据和权重不包含在仓库中。

主要默认值：200 epochs、学习率 0.02、batch size 16、训练裁剪 256、验证尺寸 320、output stride 16。根据显存调整 batch size；训练态 BatchNorm 通常要求每张 GPU 至少 2 个样本，必要时显式使用 `--freeze_bn`。

总损失为 `L_seg + alpha × L_boundary + beta × L_consistency`，默认 `alpha=0.1`、`beta=0.01`。前 5 个 epoch 使用真实可用 DEM，此后 `dem_dropout=0.2`；天然缺失样本始终走替代路径。训练集可部分缺失 DEM，但双流训练至少需要真实配对样本。

输出保存在 `run/train/exp*/`，包括配置、指标及 best/last 权重。验证和推理均对每张输入执行一次前向计算，以 `argmax` 确定每个像素的类别。指标累计整个评估集的混淆矩阵，忽略标签 255；保存的掩膜可缩放回原图大小，指标在网络输入尺寸上计算。

消融实验保持数据划分、初始化来源、输入尺寸、训练预算和随机种子一致。A/B 设置 `--beta 0 --dem_dropout 0 --alpha 0 --loss_type focal_dice`；C/D 设置 `--alpha 0 --loss_type focal_dice`；E 使用默认损失。C–E 同时报告完整 DEM 和缺失 DEM 结果，区分融合能力与缺失模态鲁棒性。增加模块并不保证指标提升，需用独立实验验证。本仓库不附带基准成绩。

## 推理

```bash
python M2FPredict_New.py --ckpt run/train/exp1/weights/best_m2f_deeplabv3plus_xception16_os16_num2.pth --input BijieLandslide/VOC2012/JPEGImages --dem_dir BijieLandslide/VOC2012/DEMImages --dem_mode auto
```

`--dem_mode auto` 自动使用匹配 DEM，缺失时走 FHM；`required` 在缺少 DEM 时报错；`missing` 强制模拟缺失。模型类型和验证尺寸默认从权重配置读取。权重须与本实现的结构匹配；加载时严格核对参数。

添加 `--compare_dem --save_probs` 可保存同一权重在真实/缺失 DEM 下的 logits、概率及掩膜差异。添加 `--gt_dir` 可对输入列表中具有对应标签的样本计算指标；正式测试应仅提供预先划定的测试样本。

可选 Grad-CAM：安装 `requirements-optional.txt`，运行 `run_gradcam.py`，其输入参数与推理脚本相同，默认 `--dem_mode required`。可设 `--dem_mode missing` 比较同一层的响应；热力图相似度不能替代模态消融实验。

## 检查与来源

```bash
python -m unittest discover -s tests -v
```

基础实现参考 [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)、[deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch) 和 [torchvision](https://github.com/pytorch/vision)。第三方许可见 `THIRD_PARTY_NOTICES.md`。
