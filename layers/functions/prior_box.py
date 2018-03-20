from __future__ import division
import torch, pdb
from math import sqrt as sqrt
from itertools import product as product
import math

def mesh(x, y):
    '''
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    '''

    # Create linear tensor from `0 - x`, and repeat it `y` times
    xs = torch.arange(0, x).repeat(y)
    # `[0,0,0,0,1,1,1,1,2,2,2,2...]`
    ys = torch.arange(0, y).view(-1, 1).repeat(1, x).view(-1)
    # stack them side by side
    return torch.stack([xs, ys], dim=1)

def hw2xy(boxes):
    a = boxes[:, :2]
    b = boxes[:, 2:]
    return torch.cat([a - b/2, a + b/2], dim=1)

def xy2hw(boxes):
    a = boxes[:, :2]
    b = boxes[:, 2:]
    return torch.cat([(a+b)/2, b - a], dim=1)

def prior_box(input_size):
    areas = [16*16, 32*32]
    aspect_ratios = [0.5, 1.0, 2.0]
    scales = [1.0, pow(2.0, 1.0/3.0), pow(2.0, 2.0/3.0)]
    if isinstance(input_size, int):
        input_size = torch.Tensor([input_size, input_size])
    else:
        input_size = torch.Tensor(input_size)

    anchor_hw = []
    for area in areas:
        for ar in aspect_ratios:
            h = math.sqrt(float(area) / ar)
            w = ar * h
            for scale in scales:
                anchor_hw.append([w * scale, h * scale])
    # Tensor: NFMs X 9 X 2
    hws = torch.Tensor(anchor_hw).view(len(areas), -1, 2)
    anchors = []
    for i, area in enumerate(areas):
        fm_size = (input_size / pow(2.0, i+2)).ceil()
        width = int(fm_size[0])
        height = int(fm_size[1])

        grid_size = input_size / width

        xy = mesh(width, height) + 0.5 # center point
        # Create 9 xy points for each point in each grid cell
        xy = (xy * grid_size).view(height, width, 1, 2).expand(height, width, 9, 2)

        wh = hws[i].view(1,1,9,2).expand(height, width, 9, 2)

        boxes = torch.cat([xy, wh], dim=3)
        anchors.append(boxes.view(-1, 4))

    result = torch.cat(anchors, dim=0)

    result = hw2xy(result)

    # result[:, (0, 2)] = result[:, (0, 2)].clamp(min=0, max=input_size[1])
    # result[:, (1, 3)] = result[:, (1, 3)].clamp(min=0, max=input_size[0])

    result = result / torch.cat([input_size, input_size], dim=0)
    result.clamp_(max=2, min=0)
    return result

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        if self.version == 'v2':
            for k, f in enumerate(self.feature_maps):
                for i, j in product(range(f), repeat=2):
                    f_k = self.image_size / self.steps[k]
                    # unit center x,y
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k]/self.image_size
                    mean += [cx, cy, s_k, s_k]

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        else:
            # original version generation of prior (default) boxes
            for i, k in enumerate(self.feature_maps):
                step_x = step_y = self.image_size/k
                for h, w in product(range(k), repeat=2):
                    c_x = ((w+0.5) * step_x)
                    c_y = ((h+0.5) * step_y)
                    c_w = c_h = self.min_sizes[i] / 2
                    s_k = self.image_size  # 300
                    # aspect_ratio: 1,
                    # size: min_size
                    mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                             (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    if self.max_sizes[i] > 0:
                        # aspect_ratio: 1
                        # size: sqrt(min_size * max_size)/2
                        c_w = c_h = sqrt(self.min_sizes[i] *
                                         self.max_sizes[i])/2
                        mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                 (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    # rest of prior boxes
                    for ar in self.aspect_ratios[i]:
                        if not (abs(ar-1) < 1e-6):
                            c_w = self.min_sizes[i] * sqrt(ar)/2
                            c_h = self.min_sizes[i] / sqrt(ar)/2
                            mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                     (c_x+c_w)/s_k, (c_y+c_h)/s_k]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
