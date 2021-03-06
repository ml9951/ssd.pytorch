"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import pdb
import os
import glob
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

class VOC(data.Dataset):
    name = 'VOC'
    def __init__(self, root_dir = '../data/VOCdevkit/VOC2007', keep_difficult=False, transform = None):
        self.transform = transform
        self.root_dir = root_dir
        anno_files = glob.glob(os.path.join(root_dir, 'Annotations/*.xml'))
        class_map = {k : i for i, k in enumerate(VOC_CLASSES)}

        self.annos = []
        for file in anno_files:
            anno = ET.fromstring(open(file).read())
            imgfile = os.path.join(root_dir, 'JPEGImages', anno.find('filename').text)
            height = float(anno.find('size').find('height').text)
            width = float(anno.find('size').find('width').text)

            current = {'img' : imgfile, 'objects' : [], 'height': height, 'width': width}

            for obj in anno.findall('object'):
                bbox = obj.find('bndbox')
                if keep_difficult or obj.find('difficult').text == '0':
                    current['objects'].append((
                        float(bbox.find('xmin').text),
                        float(bbox.find('ymin').text),
                        float(bbox.find('xmax').text),
                        float(bbox.find('ymax').text),
                        float(class_map[obj.find('name').text])
                    ))
            if len(current['objects']) > 0:
                self.annos.append(current)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        img = Image.open(anno['img'])
        target = torch.Tensor(anno['objects'])
        if self.transform:
            img, target = self.transform(img, target)
        return img, target, anno['height'], anno['width']

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    imgs, targets, heights, widths = zip(*batch)
    return torch.stack(imgs, 0), targets
    

if __name__ == '__main__':
    ds = VOC('../../data1/VOCdevkit/VOC2012')

    img, target, height, width = ds[0]
    pdb.set_trace()

