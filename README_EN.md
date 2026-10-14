# M2F-DeepLab

[简体中文](README.md) | **English**

A dual-stream landslide semantic segmentation model based on DeepLabV3+, using RGB imagery and three-channel terrain imagery.

## Model

RGB and DEM inputs have separate encoders. When DEM data is available, the fusion module uses real terrain features. When DEM data is missing, FHM generates substitute terrain features from high-level RGB features. MEF combines the two feature streams through spatial gating, channel gating, and residual connections. BEM generates boundary features in the decoder and contributes to the final segmentation.

The `has_dem` flag explicitly indicates whether each sample has terrain data. During training, some paired DEM inputs can be randomly hidden so that FHM receives segmentation supervision. Paired samples also provide feature consistency supervision. FHM detaches both its RGB input features and the DEM teacher targets from gradient computation. Real DEM features enter the fusion module directly when both modalities are available.

| Experiment | `--model` | Configuration |
| --- | --- | --- |
| A | `single_deeplabv3plus_xception16` | RGB DeepLabV3+ |
| B | `m2f_deeplabv3plus_xception16_b` | Dual encoders + high-level feature concatenation and a 1×1 projection |
| C | `m2f_deeplabv3plus_xception16_c` | B + FHM |
| D | `m2f_deeplabv3plus_xception16_d` | Dual encoders + FHM + MEF |
| E | `m2f_deeplabv3plus_xception16` | D + BEM; the default model |

A–E use the same Xception16 backbone specifications, ASPP dilation rates, and input settings. B uses zero-valued terrain features when DEM data is missing. Its concatenation operates on encoded features, rather than on a six-channel input formed by concatenating raw RGB and DEM images.

## Data and dependencies

Python 3.10+ is required. First install PyTorch and torchvision builds compatible with the CUDA environment on your training server, then install the dependencies:

```bash
pip install -r requirements.txt
```

```text
BijieLandslide/VOC2012/
├── JPEGImages/             # RGB images
├── DEMImages/              # Aligned three-channel uint8 terrain images; some may be missing
├── SegmentationClass/      # 0 = background, 1 = landslide, 255 = ignore
├── EdgeClass/              # 0 = non-boundary, 255 = boundary (converted to 0/1 when loaded)
└── ImageSets/Segmentation/
    ├── train.txt           # One sample filename or filename stem per line
    └── val.txt
```

RGB images, DEM images, segmentation masks, and boundary masks must be spatially aligned and have matching dimensions. Files are matched by filename stem; formats such as PNG, JPG, and TIF are supported. The three DEM channels represent elevation, slope, and aspect, in that order. Training and inference must use the same encoding and ImageNet mean/std normalization. `tools/prepare_dem.py` can generate an elevation/Sobel gradient magnitude/gradient direction encoding from already registered single-channel DEMs, with each channel normalized per image. It does not perform geographic registration, and its gradient channel is not a physical slope measurement calibrated to pixel size.

Generate boundary labels (all model variants use the same data interface):

```bash
python tools/Generate_Edge_From_Mask.py --mask_dir BijieLandslide/VOC2012/SegmentationClass --output_edge_dir BijieLandslide/VOC2012/EdgeClass
```

If your segmentation masks encode background/landslide as 0/255, add `--mask_encoding binary255` both when generating boundary labels and when training.

## Training and validation

The following command trains from random initialization:

```bash
python train.py --data_root BijieLandslide --no-pretrained_backbone --eval_missing_dem
```

To initialize Xception with ImageNet weights, place the corresponding weights at `weight/xception_pytorch_imagenet.pth` and omit `--no-pretrained_backbone`. Datasets and weights are not included in this repository.

The main defaults are 200 epochs, a learning rate of 0.02, a batch size of 16, a training crop size of 256, a validation size of 320, and an output stride of 16. Adjust the batch size to fit GPU memory. BatchNorm in training mode typically requires at least two samples per GPU; explicitly use `--freeze_bn` when needed.

The total loss is `L_seg + alpha × L_boundary + beta × L_consistency`, with `alpha=0.1` and `beta=0.01` by default. Available real DEM inputs are used for the first five epochs; after that, `dem_dropout=0.2` is applied. Samples that naturally lack DEM data always use the substitute-feature path. The training set may contain samples without DEMs, but dual-stream training requires at least some genuinely paired RGB/DEM samples.

Outputs are saved under `run/train/exp*/`, including the configuration, metrics, and best/last checkpoints. Validation and inference use a single forward pass per input for each evaluated modality setting, with `argmax` assigning a class to each pixel. Metrics accumulate a confusion matrix over the full evaluation set and ignore label 255. Saved masks can be resized back to the original image dimensions, while metrics are computed at the network input resolution.

For ablation experiments, keep the data split, initialization source, input size, training budget, and random seed consistent. Use `--beta 0 --dem_dropout 0 --alpha 0 --loss_type focal_dice` for A/B, `--alpha 0 --loss_type focal_dice` for C/D, and the default loss for E. For C–E, report results with both available and missing DEMs to distinguish fusion performance from robustness to missing modalities. Adding modules does not guarantee better metrics; improvements must be verified through independent experiments. This repository does not include benchmark results.

## Inference

```bash
python M2FPredict_New.py --ckpt run/train/exp1/weights/best_m2f_deeplabv3plus_xception16_os16_num2.pth --input BijieLandslide/VOC2012/JPEGImages --dem_dir BijieLandslide/VOC2012/DEMImages --dem_mode auto
```

`--dem_mode auto` uses a matching DEM when available and falls back to FHM when it is missing. `required` raises an error if a DEM is missing, while `missing` forces the missing-DEM setting. The model type and validation size are read from the checkpoint configuration by default. Checkpoints must match this implementation's architecture; parameter loading is strict.

Add `--compare_dem --save_probs` to save logits, probabilities, and mask differences between real-DEM and missing-DEM inference using the same checkpoint. Add `--gt_dir` to compute metrics for input samples with matching ground-truth masks. For formal testing, supply only the samples in your predefined test split.

For optional Grad-CAM visualization, install `requirements-optional.txt` and run `run_gradcam.py`. Its input arguments match those of the inference script, and its default is `--dem_mode required`. Use `--dem_mode missing` to compare responses from the same layer. Heatmap similarity is not a substitute for modality ablation experiments.

## Checks and acknowledgments

```bash
python -m unittest discover -s tests -v
```

The base implementation draws on [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch), [deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch), and [torchvision](https://github.com/pytorch/vision). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party license notices.
