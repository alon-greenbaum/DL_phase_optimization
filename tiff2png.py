import os
from skimage.io import imread, imsave
import numpy as np

def autonormalize(img):
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)
    img = (img * 255).astype(np.uint8)
    return img

def convert_tiffs_to_pngs(input_dir):
    png_dir = os.path.join(input_dir, "pngs")
    os.makedirs(png_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.tif', '.tiff')):
            tiff_path = os.path.join(input_dir, fname)
            img = imread(tiff_path)
            img_norm = autonormalize(img)
            png_name = os.path.splitext(fname)[0] + ".png"
            png_path = os.path.join(png_dir, png_name)
            imsave(png_path, img_norm)
            print(f"Saved {png_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python convert_tiffs_to_pngs.py <input_directory>")
        exit(1)
    convert_tiffs_to_pngs(sys.argv[1])