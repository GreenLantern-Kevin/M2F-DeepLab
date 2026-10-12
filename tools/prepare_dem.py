import os

import cv2

import numpy as np

from PIL import Image

from tqdm import tqdm

def process_and_align_dems(rgb_dir, raw_dem_dir, output_dem_dir):
    """
    对已配准且同范围的单通道 DEM 重采样，生成高程/Sobel 梯度幅度/方向三通道编码。
    """
    if not os.path.exists(output_dem_dir):
        os.makedirs(output_dem_dir)

    # 获取所有 rgb 图片的名字 (假设后缀为 .jpg)
    rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png', '.tif'))]

    print(f"开始处理 DEM 数据，总计找到 {len(rgb_files)} 个对应的影像目标...")

    for rgb_file in tqdm(rgb_files):
        stem = os.path.splitext(rgb_file)[0]
        rgb_path = os.path.join(rgb_dir, rgb_file)

        # 寻找对应的原始 DEM (支持多种常见遥感后缀)
        raw_dem_path = None
        for ext in ['.tif', '.png', '.bmp']:
            temp_path = os.path.join(raw_dem_dir, stem + ext)
            if os.path.exists(temp_path):
                raw_dem_path = temp_path
                break

        if raw_dem_path is None:
            # 说明这张图没有真实 DEM，留给 Dataset 在训练时填充零张量即可
            continue

        # 1. 读取 RGB 获取目标分辨率
        rgb_img = Image.open(rgb_path)
        target_size = rgb_img.size  # (Width, Height)

        # 2. 读取原始 DEM
        # 使用 cv2.IMREAD_UNCHANGED 防止 16-bit TIF 被强转为 8-bit
        raw_dem = cv2.imread(raw_dem_path, cv2.IMREAD_UNCHANGED)
        if raw_dem is None:
            raise ValueError(f"Unreadable DEM: {raw_dem_path}")
        if raw_dem.ndim != 2 or not np.isfinite(raw_dem).all():
            raise ValueError(f"Expected finite single-band raw DEM: {raw_dem_path}")

        # 转换为 float32 进行物理计算
        dem_float = raw_dem.astype(np.float32)

        # 3. 强制重采样 (Bicubic)，解决分辨率不匹配问题
        dem_resized = cv2.resize(dem_float, target_size, interpolation=cv2.INTER_CUBIC)

        # 4. 计算坡度 (Slope) 和 坡向 (Aspect)
        # 使用 Sobel 算子计算 X 和 Y 方向的梯度
        dx = cv2.Sobel(dem_resized, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(dem_resized, cv2.CV_32F, 0, 1, ksize=3)

        slope = np.sqrt(dx ** 2 + dy ** 2)
        aspect = np.arctan2(dy, dx)

        # 5. 归一化通道到 [0, 255] 以便保存为图片输入网络
        def normalize_channel(ch):
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min < 1e-5:
                return np.zeros_like(ch, dtype=np.uint8)
            return ((ch - ch_min) / (ch_max - ch_min) * 255.0).astype(np.uint8)

        norm_elevation = normalize_channel(dem_resized)
        norm_slope = normalize_channel(slope)
        norm_aspect = normalize_channel(aspect)

        # 6. 合成 3 通道 (H, W, 3) -> R: Elevation, G: Slope, B: Aspect
        dem_3d = np.stack([norm_elevation, norm_slope, norm_aspect], axis=-1)

        # 7. 保存到目标文件夹 (统一保存为 png 避免损失)
        save_path = os.path.join(output_dem_dir, f"{stem}.png")
        Image.fromarray(dem_3d).save(save_path)

    print("DEM 通道计算和尺寸重采样完成；地理配准须在上游完成。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--raw_dem_dir", required=True)
    parser.add_argument("--output_dem_dir", required=True)
    opts = parser.parse_args()
    process_and_align_dems(opts.rgb_dir, opts.raw_dem_dir, opts.output_dem_dir)
