import glob
import sys

from PIL import Image


def convert_png_to_jpg(pattern):
    for img_path in glob.glob(pattern):
        if img_path.endswith(".png"):
            img = Image.open(img_path)
            jpg_path = img_path[:-4] + ".jpg"

            # Convert the image to RGB mode if it's not already
            if img.mode != "RGB":
                img = img.convert("RGB")

            img.save(jpg_path, "JPEG", quality=90)
            print(f"Converted {img_path} to JPG.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python png-to-jpg.py '<pattern>'")
        sys.exit(1)

    pattern = sys.argv[1]
    convert_png_to_jpg(pattern)
