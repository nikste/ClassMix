import torch
import random
import numpy as np
import PIL
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import albumentations as A
import random
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    Rotate,
    Blur,
    RGBShift,
    Posterize,
    MotionBlur,
    RandomFog,
    Solarize,
    RandomSunFlare,
    ImageCompression,
    RandomBrightness,
    RandomContrast,
)


def horizontal_flip(v, ):
    return HorizontalFlip(p=1)


def rotate(v):
    return Rotate(limit=v, p=1)


def blur(v):
    return Blur(blur_limit=v, p=1)


def rgbshift(v):
    return RGBShift(r_shift_limit=v, b_shift_limit=v, g_shift_limit=v, p=1)


def clahe(v):
    return CLAHE(clip_limit=v, p=1)


def motion_blur(v):
    return MotionBlur(blur_limit=v, p=1)


def image_compression(v):
    v = int(v)
    return ImageCompression(quality_lower=80, quality_upper=v, p=1)


def random_brightness(v):
    return RandomBrightness(limit=v, p=1)


def random_contrast(v):
    return RandomContrast(limit=v, p=1)


def cutout(img, mask, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.shape[0]
    return cutoutabs(img, mask, v)


def cutoutabs(img, mask, v):  # [0, 60] => percentage: [0, 0.2]

    # assert 0 <= v <= 20
    if v < 0:
        return img, mask


    h = img.shape[0]
    w = img.shape[1]
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))

    color = (0, 0, 0)
    img = img.copy()
    img[y0:y1, x0:x1] = [0, 0, 0]

    mask = mask.copy()
#    mask[y0:y1, x0:x1] = 0
    return img, mask


class RandAugment(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m #[0, 30]
        self.augmentations = [(horizontal_flip, 1, 1),
                              (rotate, -20, 20),
                              (blur, 3, 7),
                              (rgbshift, -20, 20),
                              (clahe, 1, 4),
                              (motion_blur, 3, 7),
                              (image_compression, 80, 100),
                              (random_brightness, -0.2, 0.2),
                              (random_contrast, -0.2, 0.2),
                              (cutout, 0, 0.2)
                              ]

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        ops = random.choices(self.augmentations, k=self.n)
        for op, min_val, max_val in ops:
            val = (float(self.m) / 30) * float(max_val - min_val) + min_val
            if op == cutout:
                img, mask = cutout(img, mask, val)
            else:
                aug = op(val)
                augmented = aug(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
        return {'image': img, 'label': mask}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}

class Resize(object):
    def __init__(self, resize_size):
        self.resize_height = resize_size[0]
        self.resize_width = resize_size[1]
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        image_PIL = Image.fromarray(img)
        label_PIL = Image.fromarray(mask)
        image = transforms.functional.resize(image_PIL, (self.resize_height, self.resize_width))
        label = transforms.functional.resize(label_PIL, (self.resize_height, self.resize_width), interpolation=Image.NEAREST)
        label = np.array(label, dtype=np.uint8)
        image = np.array(image, dtype=np.uint8)
        return {'image': image.copy(), 'label': label.copy()}


class LabelMapping(object):
    def __init__(self, label_mapping):
        self.label_mapping = label_mapping

    def __call__(self, sample):
        label = np.array(sample['label'], dtype=np.uint8)
        label_remap = np.ones(label.shape, np.uint8)
        for original_label, mapped_label in self.label_mapping.items():
            mask = label == original_label
            label_remap[mask] = mapped_label
        return {'image': sample['image'].copy(),
                'label': label_remap.copy()}


class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        image_PIL = Image.fromarray(img)
        label_PIL = Image.fromarray(mask)
        i, j, h, w = transforms.RandomCrop.get_params(
            image_PIL, output_size=(self.crop_height, self.crop_width))
        image = transforms.functional.crop(image_PIL, i, j, h, w)
        label = transforms.functional.crop(label_PIL, i, j, h, w)
        label = np.array(label, dtype=np.uint8)
        image = np.array(image, dtype=np.uint8)

        return {'image': image.copy(), 'label': label.copy()}


