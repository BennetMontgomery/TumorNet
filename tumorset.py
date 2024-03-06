import copy
import os
import json
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import Grayscale
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset
from matplotlib.path import Path

class TumorSet(Dataset):
    def __init__(self, img_dir, annotations_file='_annotations.coco.json', transform=Grayscale(), target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # import data from annotations file
        data_dict = json.load(open(os.path.join(img_dir, annotations_file), 'r'))
        self.annotations = copy.deepcopy(data_dict['annotations'])
        self.image_data = copy.deepcopy(data_dict['images'])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        # load image from file
        image = read_image(str(os.path.join(self.img_dir, self.image_data[idx]['file_name'])))
        image = self.transform(image)
        image = convert_image_dtype(image)

        # generate segmentation from annotations matching the image id
        segmentations = []
        for annotation in self.annotations:
            if annotation['image_id'] == self.image_data[idx]['id'] and annotation['category_id'] == 2:
                segmentations.append(annotation['segmentation'][0])

        grid = np.array([[False for _ in range(self.image_data[idx]['width'])] for _ in range(self.image_data[idx]['height'])])
        for segmentation in segmentations:
            vertices = list(zip(segmentation[::2], segmentation[1::2]))
            x,y = np.meshgrid(np.arange(self.image_data[idx]['width']), np.arange(self.image_data[idx]['height']))
            x,y = x.flatten(), y.flatten()

            points = np.vstack((x,y)).T

            path = Path(vertices)
            new_grid = path.contains_points(points)
            grid = grid | new_grid.reshape((self.image_data[idx]['width'], self.image_data[idx]['height']))

        grid = [grid]
        target_grid = np.where(grid, 1, 0)
        target_grid_tensor = torch.from_numpy(target_grid)

        # return the image tensor and segmentation grid
        return image, target_grid_tensor