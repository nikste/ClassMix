import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils import data
from PIL import Image
import glob
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import transformations as tr



class Nemesis():
    def __init__(self, root, training=True, image_scaling_ratio=None):
        self.root = root
        self.files = []
        self.training = training
        self.image_scaling_ratio = image_scaling_ratio
        print("going through root nemesis", root)
        all_img_files = [f for f in Path(root).glob('**/*')]
        print("found nemesis:", len(all_img_files))
        for f in all_img_files:
            self.files.append({"img": str(f)})
        #for dir in [f for f in Path(root):
        #    print("nemesis, looking into", osp.join(root, 'images'), "found", len(list(glob.glob(osp.join(root, 'images/*.jpg')))))
        #    for file_path in glob.glob(osp.join(dir, 'images/*.jpg')):
        #        filename = osp.basename(file_path).split('.')[0]
        #        img_file = file_path
        #        print("appending", img_file)
        #        self.files.append({
        #            "img": img_file,
        #        })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.zeros_like(image)
        if self.image_scaling_ratio:
            image = cv2.resize(image, (0, 0), fx=self.image_scaling_ratio, fy=self.image_scaling_ratio)
            label = cv2.resize(label, (0, 0), fx=self.image_scaling_ratio, fy=self.image_scaling_ratio,
                               interpolation=cv2.INTER_NEAREST)

        sample = {"image": image, "label": label}
        sample = self.transforms_tr(sample)
        return sample["image"], sample["label"], -1, -1, -1 # dummy values
        # return sample

    def transforms_tr(self, sample):
        if sample['image'].shape[0] < 720 or sample['image'].shape[1] < 960:
            composed_transforms = transforms.Compose([
                tr.Resize((720, 960)),
                tr.ToTensor()])

        else:
            composed_transforms = transforms.Compose([
                tr.RandomCrop((720, 960)),
                tr.ToTensor()])
        return composed_transforms(sample)
if __name__ == "__main__":

    # from dataloaders import custom_transforms as tr
    # from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        help="Path to the directory containing the PASCAL VOC dataset.")

    args = parser.parse_args()
    train_dataset = Nemesis(args.data_dir)
    trainloader = data.DataLoader(train_dataset, shuffle=True, batch_size=1,
                                  num_workers=4, pin_memory=True)
    cv2.namedWindow("image")
    trainloader_enu = enumerate(trainloader)

    for step in range(10):
        batch = next(trainloader_enu)
        index, sample = batch
        image = sample["image"]
        image = np.array(image[0]).astype(np.uint8)
        image = image.transpose((1, 2, 0))
        cv2.imshow("image", image)
        cv2.waitKey()
