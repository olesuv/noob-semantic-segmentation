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

    def get_train_img_id_filename(self):
        self.IMG = []
        with open('./data/train/_annotations.coco.json') as f:
            train_json = json.load(f)
            for img in train_json['images']:
                self.IMG.append({"id": img['id'], "file_name": img['file_name']})
        return self.IMG

    def get_train_annotations(self):
        self.ANN = []
        with open('./data/train/_annotations.coco.json') as f:
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

            x, y, w, h = map(int, map(round, bbox))
        
            x_end = min(x + w, img.shape[1])
            y_end = min(y + h, img.shape[0])

            mask = np.zeros_like(img)
            mask[y:y_end, x:x_end] = (255, 255, 255)

            mask_path = os.path.join(output_dir, img_info['file_name'])
            cv2.imwrite(mask_path, mask)
            masks_n += 1

        print(f"Generated {masks_n} masks")

    def resize_img_dir_128(self, img_dir='./data/train'):
        for filename in os.listdir(img_dir):
            if filename.endswith('.json'):
                continue
            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, (128, 128))
                cv2.imwrite(img_path, resized_img)
            else:
                print(f"Cannot read image file {img_path}. Skipping.")
        print(f"Resized directory with name `{img_dir}` to 128x128 images")

data_cleaner = DataClean()
data_cleaner.get_train_img_id_filename()
data_cleaner.get_train_annotations()
data_cleaner.merge_img_ann()
data_cleaner.generate_masked_imgs()

