"""SpaceNet Dataset

Author: Matt Le

"""

import pdb, os, glob, sys, torch, numpy as np, json
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

class SpaceNet(data.Dataset):
    name = 'VOC'
    classes = ['building']
    def __init__(self, anno_file, transform = None):
        self.transform = transform
        self.root_dir = os.path.dirname(os.path.realpath(anno_file))

        self.annos = json.load(open(anno_file, 'r'))
        self.annos = list(filter(lambda x: len(x['rects']) > 0, self.annos))

        self.keys = ['x1', 'y1', 'x2', 'y2']

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        img = Image.open(os.path.join(self.root_dir, anno['image_path']))

        target = torch.Tensor([[r[k] for k in self.keys] + [0] for r in anno['rects']])

        mask = (target[:, 2] - target[:, 0] >= 3) & (target[:, 3] - target[:, 1] >= 3)

        if np.random.randint(0, 16) == 0:
            mask[:] = False


        target = target[mask.unsqueeze(1).expand_as(target)].view(-1, 5)

        if len(target) == 0:
            
            target = torch.Tensor([[2, 2, 3, 3, -1]])



        if self.transform:
            img_data, target = self.transform(img, target)

        return img_data, target, img.height, img.width

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

