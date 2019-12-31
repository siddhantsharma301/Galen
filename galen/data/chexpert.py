"""
CheXpert Dataset for training
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import cv2
import numpy as np
import torch
import pandas as pd
import torch.utils.data as data_utils
from galen.utils import image as image_utils


class CheXpertDataset(data_utils.Dataset):
    """
    CheXpert dataset class for Galen training
    """
    def __init__(self, root_dir, csv, transform=None):
        self.root_dir = root_dir
        self.csv = pd.read_csv(csv)
        self.preprocess_csv()
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.root_dir + self.csv['Path'].loc[idx]
        img = cv2.imread(path)
        img = self.preprocess_image(img)
        if self.transform:
            img = self.transform(img)
        return img

    def preprocess_csv(self):
        """
        Preprocessor for CheXpert dataset CSV
        Removes all blanks and replaces with `np.uint8`

        Args:

        Returns:

        """
        self.csv['AP/PA'] = self.csv['AP/PA'].replace(np.nan, 0)
        self.csv['No Finding'] = self.csv['No Finding'].replace(np.nan, 0).astype(np.int8)
        self.csv['Enlarged Cardiomediastinum'] = self.csv['Enlarged Cardiomediastinum'].replace(np.nan, 0).astype(np.int8)
        self.csv['Cardiomegaly'] = self.csv['Cardiomegaly'].replace(np.nan, 0).astype(np.int8)
        self.csv['Lung Opacity'] = self.csv['Lung Opacity'].replace(np.nan, 0).astype(np.int8)
        self.csv['Lung Lesion'] = self.csv['Lung Lesion'].replace(np.nan, 0).astype(np.int8)
        self.csv['Edema'] = self.csv['Edema'].replace(np.nan, 0).astype(np.int8)
        self.csv['Consolidation'] = self.csv['Consolidation'].replace(np.nan, 0).astype(np.int8)
        self.csv['Pneumonia'] = self.csv['Pneumonia'].replace(np.nan, 0).astype(np.int8)
        self.csv['Atelectasis'] = self.csv['Atelectasis'].replace(np.nan, 0).astype(np.int8)
        self.csv['Pneumothorax'] = self.csv['Pneumothorax'].replace(np.nan, 0).astype(np.int8)
        self.csv['Pleural Effusion'] = self.csv['Pleural Effusion'].replace(np.nan, 0).astype(np.int8)
        self.csv['Pleural Other'] = self.csv['Pleural Other'].replace(np.nan, 0).astype(np.int8)
        self.csv['Fracture'] = self.csv['Fracture'].replace(np.nan, 0).astype(np.int8)
        self.csv['Support Devices'] = self.csv['Support Devices'].replace(np.nan, 0).astype(np.int8)
        self.csv = self.csv[(self.csv['Frontal/Lateral'] == 'Frontal') & (self.csv['No Finding'] == 1) & (self.csv['Fracture'] == 0)].reset_index()

    def preprocess_image(self, img):
        """
        Preprocess each training image

        Args:
        * img
            image (`np.ndarray`)

        Returns:
        processed image (`torch.Tensor`)
        """
        return image_utils.img_to_tensor(img)


if __name__ == "__main__":
    # Check if everything works
    ROOT = "path/to/root/"
    CSV = ROOT + "CheXpert-v1.0-small/train.csv"
    dataset = CheXpertDataset(ROOT, CSV)
    print("Number of training images: " + str(len(dataset)))
    dataloader = data_utils.DataLoader(dataset, batch_size = 4)
    batch = next(iter(dataloader))
    print("Number of training batches: " + str(len(dataloader)))
    print("Image type: " + str(type(batch[0])))
    print("Done!")
    
        
