import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from dotenv import load_dotenv

# structure should be:
# .../COCO_DATA_DIR/
# ├── annotations/
# │   ├── instances_train2017.json
# │   └── instances_val2017.json
# ├── train2017/
# ├── masks_train2017/
# ├── val2017/
# └── masks_val2017/

# config
train = True
if train:
    data_type = 'train2017'
else:
    data_type = 'val_2017'

data_dir = os.environ['COCO_DATA_DIR']
ann_file = os.path.join(data_dir, 'annotations', f'instances_{data_type}.json')
output_mask_dir = os.path.join(data_dir, f'masks_{data_type}')

os.makedirs(output_mask_dir, exist_ok=True)

def main():
    # coco init to get instance annotations
    coco = COCO(ann_file)

    # get all image ids
    img_ids = coco.getImgIds()

    print(f"Found {len(img_ids)} images. Starting mask generation.")

    for img_id in tqdm(img_ids[:10]):
        # get image info
        img_info = coco.loadImgs(img_id)[0]
        img_height = img_info['height']
        img_width = img_info['width']
        img_filename = img_info['file_name']

        # get all annotation ids for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        if not ann_ids:
            continue

        anns = coco.loadAnns(ann_ids)

        # single binary mask by combining all instance masks
        combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for ann in anns:
            instance_mask = coco.annToMask(ann)
            combined_mask = np.maximum(combined_mask, instance_mask)

        # save as grayscale
        mask_image = Image.fromarray(combined_mask * 255)
        mask_image.save(os.path.join(output_mask_dir, img_filename.replace('.jpg', '.png')))

    print("Mask generation complete!!")