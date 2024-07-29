import numpy as np
import zipfile
import json
import cv2
import os


class DataClean:
    def __init__(self, data_file='./archive.zip', output_dir='./data'):
        try:
            if os.listdir(output_dir):
                return
            with zipfile.ZipFile(data_file, 'r') as zip_dir:
                zip_dir.extractall(output_dir)
        except Exception as e:
            print(f"Already extracted or no data: {e}")

    def get_img_id_filename(self, img_dir='./data/train'):
        self.IMG = []
        with open(f"{img_dir}/_annotations.coco.json") as f:
            train_json = json.load(f)
            for img in train_json['images']:
                self.IMG.append({"id": img['id'], "file_name": img['file_name']})
        return self.IMG

    def get_annotations(self, img_dir='./data/train'):
        self.ANN = []
        with open(f"{img_dir}/_annotations.coco.json") as f:
            train_json = json.load(f)
            for ann in train_json['annotations']:
                self.ANN.append({
                    "img_id": ann['image_id'],
                    "bbox": ann['bbox'],
                    "seg": ann['segmentation']
                })
        return self.ANN

    def get_filename_by_id(self, ann_img_id):
        for img in self.IMG:
            if img['id'] == ann_img_id:
                return img['file_name']
        return None

    def merge_img_ann(self):
        self.MERGE = []
        for ann in self.ANN:
            file_name = self.get_filename_by_id(ann['img_id'])
            if file_name:
                merged_item = ann.copy()
                merged_item["file_name"] = file_name
                self.MERGE.append(merged_item)
        return self.MERGE

    def generate_masked_imgs(self, img_dir='./data/train', output_dir='./data/train-masks'):
        masks_n = 0
        print('Generating masks...')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for img_info in self.MERGE:
            img_path = os.path.join(img_dir, img_info['file_name'])
            img = cv2.imread(img_path)

            if img is None:
                print(f"Cannot read image file {img_path}. Skipping.")
                continue

            bbox = img_info['bbox']
            mask = np.zeros_like(img)

            x, y, w, h = map(int, map(round, bbox))
            mask[y:y+h, x:x+w] = (255, 255, 255)

            mask_path = os.path.join(output_dir, img_info['file_name'])
            cv2.imwrite(mask_path, mask)
            masks_n += 1

        print(f"Generated {masks_n} masks")


train_images_path = './data/train'
train_masks_path = './data/train-masks'

train_dc = DataClean()
train_dc.get_img_id_filename()
train_dc.get_annotations()
train_dc.merge_img_ann()
train_dc.generate_masked_imgs()

test_images_path = './data/test'
test_masks_path = './data/test-masks'

test_dc = DataClean()
test_dc.get_img_id_filename(img_dir=test_images_path)
test_dc.get_annotations(img_dir=test_images_path)
test_dc.merge_img_ann()
test_dc.generate_masked_imgs(img_dir=test_images_path, output_dir=test_masks_path)

