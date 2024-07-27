import pandas as pd
import numpy as np
import zipfile
import json
import cv2

from os import listdir, makedirs, path


class DataClean:
    def __init__(self, data_file='./archive.zip', output_dir='./data'):
        try:
            if listdir('./date'):
                return
            with zipfile.ZipFile(data_file, 'r') as zip_dir:
                zip_dir.extractall(output_dir)
        except:
            print("Already extracted or no data")

    def get_train_img_id_filename(self):
        self.IMG = []
        with open('./data/train/_annotations.coco.json') as f:
            train_json = json.load(f)
            for img in train_json['images']:
                self.IMG.append(
                    {"id": img['id'], "file_name": img['file_name']})
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

    def merge_img_ann(self):
        self.MERGE = []
        for ann in self.ANN:
            file_name = self.get_filename_by_id(ann['img_id'])
            if file_name:
                merged_item = ann.copy()
                merged_item["file_name"] = file_name
                self.MERGE.append(merged_item)
        return self.MERGE

    def merge_to_df(self):
        res = pd.DataFrame(self.MERGE)
        return res

    def generate_masked_imgs(self, img_dir='./data/train', output_dir='./data/train-masks'):
        masks_n = 0
        print('Generating masks...')

        if not path.exists(output_dir):
            makedirs(output_dir)

        for img_info in self.MERGE:
            bbox = img_info['bbox']
            seg = img_info['seg']

            img = cv2.imread(f"{img_dir}/{img_info['file_name']}")
            mask = np.zeros_like(img)

            x, y, w, h = map(int, map(round, bbox))

            mask[y:y+h, x:x+w] = (255, 255, 255)
            result_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            cv2.imwrite(f"{output_dir}/{img_info['file_name']}", result_rgb)
            masks_n += 1

        print(f"Generated {masks_n} masks")


data_cleaner = DataClean()
data_cleaner.get_train_img_id_filename()
data_cleaner.get_train_annotations()
data_cleaner.merge_img_ann()
# print(data_cleaner.merge_to_df())
data_cleaner.generate_masked_imgs()
