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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import transformations as tr




class Blurring():
    def __init__(self, root, training=True):
        self.root = root
        self.files = []
        self.training = training
        print("blurring, looking into", osp.join(root, 'images'), "found", len(list(glob.glob(osp.join(root, 'images/*.jpg')))))
        for file_path in glob.glob(osp.join(root, 'images/*.jpg')):
            filename = osp.basename(file_path).split('.')[0]
            img_file = file_path
            label_file = osp.join(root, 'labels', filename + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = np.uint8(label)
        sample = {"image": image, "label": label}


        # image_PIL = Image.fromarray(image)
        # label_PIL = Image.fromarray(label)
        if self.training:
            sample = self.transforms_tr(sample)
        else:
            sample = self.transforms_valid(sample)
        return sample["image"], sample["label"], -1, -1, -1 # dummy values

    def transforms_tr(self, sample):
        prob_augmentation = random.random()
        # if prob_augmentation > 0.5:
        #     augmentation_strength = int(np.random.uniform(10, 30))
        #     composed_transforms = transforms.Compose([
        #         tr.RandomCrop((720, 960)),
        #         tr.RandAugment(3, augmentation_strength),
        #         tr.ToTensor()])
        # else:
        composed_transforms = transforms.Compose([
            tr.RandomCrop((720, 960)),#tr.RandomCrop((720, 960)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transforms_valid(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomCrop((720, 960)),#tr.Resize((720, 960)),
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
    train_dataset = Blurring(os.path.join(args.data_dir, 'train'))
    trainloader = data.DataLoader(train_dataset, shuffle=True, batch_size=1,
                                  num_workers=4, pin_memory=True)
    cv2.namedWindow("image")
    # cv2.namedWindow("labels")
    trainloader_enu = enumerate(trainloader)

    for step in range(10):

        batch = next(trainloader_enu)

        index, sample = batch

        label = sample["label"]
        image = sample["image"]


        # if index % 100 == 0:
        #     print('%d processd' % (index))

        # image, label, size = batch
        image = np.array(image[0]).astype(np.uint8)
        label = np.array(label[0]).astype(np.uint8)

        image = image.transpose((1, 2, 0))
        cv2.imshow("image", image)
        # # cv2.imshow("labels", label)
        #
        cv2.waitKey()
        # count += 1

