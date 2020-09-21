"""
Written by dev-kim
kim1102@kist.re.kr
2020.09.20
"""
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
import cv2

class emotion_data(Dataset):
    def __init__(self, neutral, smile):

        self. transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(200),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

        self.info_list = []

        # neutral, label:0
        n_file_list = os.listdir(neutral)
        for file in n_file_list:
            file_path = os.path.join(neutral, file)
            raw_img = cv2.imread(file_path)
            self.info_list.append({'label': 0, 'cv_img':raw_img})

        # neutral, label:1
        s_file_list = os.listdir(smile)
        for file in s_file_list:
            file_path = os.path.join(smile, file)
            raw_img = cv2.imread(file_path)
            self.info_list.append({'label': 1, 'cv_img': raw_img})

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        label = self.info_list[idx]['label']
        raw_img = self.info_list[idx]['cv_img']

        #cv2.imshow("result", im_stacked_rgb)
        #cv2.waitKey(0)

        tensor = self.transformations(raw_img)

        return label, tensor