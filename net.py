import os, pdb, numpy as np, time, json, pandas, glob, hashlib, torch
from shapely.geometry import MultiPolygon, box
from subprocess import check_output
from zipfile import ZipFile
from retina import Retina
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

pandas.options.mode.chained_assignment = None

NAME = 'SSD'

class SSD:
    name = NAME

    @classmethod
    def mk_hash(cls, path):
        '''
        Create an MD5 hash from a models weight file.
        Arguments:
            path : str - path to TensorBox checkpoint
        '''
        dirs = path.split('/')
        if 'ssd.pytorch' in dirs:
            dirs = dirs[dirs.index('ssd.pytorch'):]
            path = '/'.join(dirs)
        else:
            path = os.path.join('ssd.pytorch', path)

        md5 = hashlib.md5()
        md5.update(path.encode('utf-8'))
        return md5.hexdigest()

    @classmethod
    def zip_weights(cls, path, base_dir='./'):
        if os.path.splitext(path)[1] != '.pth':
            raise ValueError('Invalid checkpoint')

        dirs = path.split('/')

        res = {
            'name' : 'TensorBox',
            'instance' : '_'.join(dirs[-2:]),
            'id' : cls.mk_hash(path)
        }

        zipfile = os.path.join(base_dir, res['id'] + '.zip')

        if os.path.exists(zipfile):
            os.remove(zipfile)

        weight_dir = os.path.dirname(path)

        with ZipFile(zipfile, 'w') as z:
            z.write(path, os.path.join(res['id'], os.path.basename(file)))

        return zipfile

    def __init__(self, weights, classes=['building']):
        self.net = Retina(classes).eval().cuda()
        chkpnt = torch.load(weights)
        self.net.load_state_dict(chkpnt['state_dict'])
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image, threshold, eval_mode = False):
        """
        Infer buildings for a single image.
        Inputs:
            image :: n x m x 3 ndarray - Should be in RGB format
        """

        t0 = time.time()
        img = self.transform(image)
        out = self.net(Variable(img.unsqueeze(0).cuda(), volatile=True)).squeeze().data.cpu()
        total_time = time.time() - t0
        
        scores = out[:, :, 0] # class X top K X (score, minx, miny, maxx, maxy)
        
        max_scores, inds = scores.max(dim=0)

        linear = torch.arange(0, out.shape[1]).long()
        boxes = out[inds, linear].numpy()
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)] * image.width, a_min=0, a_max=image.width)
        boxes[:, (2, 4)] = np.clip(boxes[:, (2, 4)] * image.height, a_min=0, a_max=image.height)

        df = pandas.DataFrame(boxes, columns=['score', 'x1' ,'y1', 'x2', 'y2'])

        if eval_mode:
            return df[df['score'] > threshold], df, total_time
        else:
            return df[df['score'] > threshold]

        pdb.set_trace()

        

    def predict_all(self, test_boxes_file, threshold, data_dir = None):
        test_boxes = json.load(open(test_boxes_file))
        true_annolist = al.parse(test_boxes_file)
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(test_boxes_file))
        
        total_time = 0.0

        for i in range(len(true_annolist)):
            true_anno = true_annolist[i]

            orig_img = imread('%s/%s' % (data_dir, true_anno.imageName))[:,:,:3]

            pred, all_rects, time = self.predict_image(orig_img, threshold, eval_mode = True)

            pred['image_id'] = i
            all_rects['image_id'] = i

            yield pred, all_rects, test_boxes[i]


if __name__ == '__main__':
    from data import VOC_CLASSES
    img = Image.open('2007_001027.jpg')
    ssd = SSD('weights/ssd_2018-03-15T17:03:22.140268/epoch_26.pth', VOC_CLASSES)
    ssd.predict_image(img, 0.2)
    pdb.set_trace()


