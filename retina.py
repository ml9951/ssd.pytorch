from torchvision.models import resnet101
import pdb, torch
import torch.nn.functional as F
from layers import prior_box, Detect

class Retina(torch.nn.Module):
    ANCHORS_PER_GRID_CELL = 9

    def mk_subnet(self, out_factor, include_sigmoid = True):
        layers = []

        for i in range(4):
            layers.append(torch.nn.Conv2d(256, 256, 3, padding=1))
            layers.append(torch.nn.ELU(inplace=True))

        layers.append(torch.nn.Conv2d(256, self.ANCHORS_PER_GRID_CELL * out_factor, 3, padding=1))
        if include_sigmoid:
            layers.append(torch.nn.Sigmoid())
        return torch.nn.Sequential(*layers)

    def init_weights(self):
        stack = list(self.named_children())
        while len(stack) > 0:
            name, layer = stack.pop()
            if isinstance(layer, torch.nn.Sequential) and name != '_backbone':
                stack.extend(list(layer.named_children()))
            elif isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.xavier_normal(layer.weight)
                layer.bias.data.zero_()

    def __init__(self, classes, size):
        super(Retina, self).__init__()

        self.size = size

        self.priors = torch.autograd.Variable(prior_box(size), requires_grad=False)

        mask = ((self.priors[:,2] > self.priors[:,0]) & (self.priors[:,3] > self.priors[:,1]))

        self.classes = classes
        self.num_classes = len(classes) + 1

        self._backbone = resnet101(pretrained=True)

        names, layers = zip(*list(self._backbone.named_children())[:-2]) # leave off avgpool and fc

        self.backbone = []

        i = 0
        while i < len(names):
            j = i + 1
            while j < len(names) and not(names[j].startswith('layer')):
                j += 1
            self.backbone.append(torch.nn.Sequential(*layers[i:j]))
            i = j

        self.conv6 = torch.nn.Conv2d(2048, 256, 3, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(2048, 256, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(1024, 256, 1)
        self.conv3 = torch.nn.Conv2d(512, 256, 1)
        self.conv2 = torch.nn.Conv2d(256, 256, 1)

        self.loc = self.mk_subnet(4, include_sigmoid=False)
        self.conf = self.mk_subnet(self.num_classes, include_sigmoid = False)

        self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # Pyramid
        c1 = self.backbone[0](x)
        c2 = self.backbone[1](c1)
        c3 = self.backbone[2](c2)
        c4 = self.backbone[3](c3)
        c5 = self.backbone[4](c4)

        # According to the paper: 
        # RetinaNet uses feature pyramid levels P3 to P7, where P3 to P5 are
        # computed from the output of the corresponding ResNet residual stage (C3
        # through C5) using top-down and lateral connections just as in [19], P6 is
        # obtained via a 3×3 stride-2 conv on C5, and P7 is computed by applying
        # ReLU followed by a 3×3 stride-2 conv on P6. 
        p6 = self.conv6(c5)
        p7 = F.relu(self.conv7(p6))
        p5 = self.conv5(c5)

        # bilinear upsampling allows for upsampling to arbitrary size in case the input image
        # is not of a size that is divisible by a power of 2
        p4 = F.upsample(p5, size=tuple(c4.shape[2:4]), mode='bilinear') + self.conv4(c4)
        p3 = F.upsample(p4, size=tuple(c3.shape[2:4]), mode='bilinear') + self.conv3(c3)
        p2 = F.upsample(p3, size=tuple(c2.shape[2:4]), mode='bilinear') + self.conv2(c2)

        loc_pred, conf_pred = [], []

        # Localization/Classification
        for fm in [p2, p3]: #, p4, p5, p6, p7]:
            locs = self.loc(fm).permute((0, 2, 3, 1))
            confs = self.conf(fm).permute((0, 2, 3, 1))

            # reshape from two dimensional matrix of boxes, to linear layout
            locs = locs.contiguous().view(batch_size, -1, 4)
            confs = confs.contiguous().view(batch_size, -1, self.num_classes)

            loc_pred.append(locs)
            conf_pred.append(confs)

        loc = torch.cat(loc_pred, dim=1)
        conf = torch.cat(conf_pred, dim=1)

        if not self.training:
            return self.detect(
                loc,                                # loc preds
                F.softmax(conf, dim=2),                 # conf preds
                self.priors.type(type(x.data))      # default boxes
            )
        else:
            return (loc, conf, self.priors)

if __name__ == '__main__':
    net = Retina(['background', 'building'])
    net(torch.autograd.Variable(torch.rand(1, 3, 300, 300)))
