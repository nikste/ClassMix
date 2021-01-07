import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.voc_dataset import VOCDataSet

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "pascal_voc": VOCDataSet
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return './data/CityScapes/'
    elif name == 'pascal_voc':
        return './data/VOC2012/'
    elif name == 'semantic_kitti':
        return './data/semantic_kitti'

