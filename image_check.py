import os
from PIL import Image
import glob
import numpy as np

def get_image_sizes(folder_path, extensions=('*.jpg', '*.png', '*.jpeg')):
    """
    统计文件夹中所有图片的尺寸。
    
    Args:
        folder_path (str): 图片文件夹路径
        extensions (tuple): 要统计的图片文件扩展名
    
    Returns:
        list: 包含每个图片尺寸的列表 [(width, height), ...]
    """
    image_sizes = []
    # 获取所有指定扩展名的图片文件
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                image_sizes.append((width, height))
                print(f"Image: {os.path.basename(img_path)}, Size: {width}x{height}")
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    return image_sizes

def summarize_image_sizes(image_sizes):
    """
    汇总图片尺寸的统计信息。
    
    Args:
        image_sizes (list): 包含图片尺寸的列表 [(width, height), ...]
    """
    if not image_sizes:
        print("No valid images found.")
        return
    
    # 转换为 numpy 数组便于统计
    sizes = np.array(image_sizes)
    widths = sizes[:, 0]
    heights = sizes[:, 1]
    
    # 统计信息
    print("\nImage Size Statistics:")
    print(f"Total images: {len(image_sizes)}")
    print(f"Max width: {widths.max()}, Max height: {heights.max()}")
    print(f"Min width: {widths.min()}, Min height: {heights.min()}")
    print(f"Average width: {widths.mean():.2f}, Average height: {heights.mean():.2f}")
    print(f"Median width: {np.median(widths):.2f}, Median height: {np.median(heights):.2f}")

if __name__ == '__main__':
    # 指定图片文件夹路径
    folder_path = './data/image/HarMeme/All'  # 替换为你的图片文件夹路径
    image_sizes = get_image_sizes(folder_path)
    summarize_image_sizes(image_sizes)