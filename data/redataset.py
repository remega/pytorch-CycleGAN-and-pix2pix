import torchvision.transforms as Trans
import os
import glob
import random
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
import torch.utils.data as data
from PIL import Image
#from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random

class dataset_sal(BaseDataset):
  def __init__(self, opts):
    self.opt = opts
    self.oriimg = []
    self.orisal = []
    for dirname in opts.datasets:
      tempdir = os.path.join(opts.dataroot, dirname)
      images = os.listdir(tempdir)
      for x in images:
        curdir = os.path.join(tempdir, x)
        if os.path.isdir(curdir):
          self.oriimg = self.oriimg + [os.path.join(tempdir, x, 'Ori.png')]
          self.orisal = self.orisal + [os.path.join(tempdir, x, 'SalmapOri.png')]
    self.size = len(self.oriimg)
    # setup image transformation
    # transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    # transform_list += [Trans.ToTensor()]
    # transform_list += [Trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    # self.transforms_img = Trans.Compose(transform_list)
    # transform_list = [Trans.Resize((opts.salmap_resize_size, opts.salmap_resize_size), Image.BICUBIC)]
    # transform_list += [Trans.ToTensor()]
    # self.transforms_salmap = Trans.Compose(transform_list)
    # transform_list = [Trans.Resize((opts.img_resize_size, opts.img_resize_size), Image.BICUBIC)]
    # transform_list += [Trans.ToTensor()]
    # self.transforms_salmapup = Trans.Compose(transform_list)
    print('validation %s: %d images'%(opts.dataroot, self.size))

  def __getitem__(self, index):
    img = Image.open(self.oriimg[index]).convert('RGB')
    tempmap = Image.open(self.orisal[index])

    transform_params = get_params(self.opt, tempmap.size)
    A_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.input_nc == 1))
    B_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.output_nc == 1))

    A = A_transform(tempmap)
    B = B_transform(img)

    return {'A': A, 'B': B, 'A_paths': self.orisal[index], 'B_paths': self.oriimg[index]}

  def __len__(self):
    return self.size

