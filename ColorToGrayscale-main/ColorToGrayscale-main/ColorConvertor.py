from fastai.vision.all import *
from pathlib import Path
import cv2

path = Path(r"C:\Users\afnan\OneDrive\Pictures\Camera Roll\ColorizationDataset") #path to your folder

(path/"grayscale").mkdir(exist_ok=True)

color_images = get_image_files(path/"color")

for img_path in color_images:
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(path/"grayscale"/img_path.name), gray)

print(f"Converted {len(color_images)} images to grayscale.")

assert (path/"grayscale").exists(), "'grayscale' folder not found!" #path to your grayscale images folder
assert (path/"color").exists(), "'color' folder not found!" #path to your output or color images folder

dblock = DataBlock(
    blocks=(ImageBlock, ImageBlock),
    get_items=get_image_files,
    get_y=lambda x: path/"color"/x.name,
    splitter=RandomSplitter(valid_pct=0.4, seed=42),
    item_tfms=Resize(256)
)

dls = dblock.dataloaders(path/"grayscale", bs=4)

dls.show_batch()
