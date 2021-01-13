from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree as kdtree

splits = {
    "train": [1, 2, 0, 3, 4, 5, 6, 7, 9, 10],
    "val": [8],
    "test": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}


def _transorm_train(depth, refl, labels_image, labels, py, px, points_xyz, points_refl, new_h=289, new_w=4097, offset_augmentation=True):
    # new_h = 145 #289
    # new_w = 2046 #4097

    py = new_h * py / 65.0
    px = new_w * px / 2049.0

    depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    refl = cv2.resize(refl, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    labels_image = cv2.resize(labels_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if offset_augmentation:
        offset_x = np.random.randint(depth.shape[1] - 2049 + 1)#1025 + 1)
        offset_y = 0 #np.random.randint(depth.shape[0] - 289 + 1)

        depth = depth[offset_y : offset_y + new_h, offset_x : offset_x + 2049]#1025]
        refl = refl[offset_y : offset_y + new_h, offset_x : offset_x + 2049]#1025]
        labels_image = labels_image[offset_y : offset_y + new_h, offset_x : offset_x + 2049]#1025]

        py = (py - offset_y) / new_h
        px = (px - offset_x) / 2049 #1025

    valid = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)
    labels = labels[valid]
    points_refl = points_refl[valid]
    px = px[valid]
    py = py[valid]
    points_xyz = points_xyz[valid, :]
    px = 2.0 * (px - 0.5)
    py = 2.0 * (py - 0.5)

    if np.random.uniform() > 0.5:
        depth = np.flip(depth, axis=1).copy()
        refl = np.flip(refl, axis=1).copy()
        labels_image = np.flip(labels_image, axis=1).copy()
        px *= -1

    min_n_points = int(38_000 * 289 / new_h)
    if px.shape[0] < min_n_points:
        pad_len = min_n_points - px.shape[0]
        px = np.hstack([px, np.zeros((pad_len,))])
        py = np.hstack([py, np.zeros((pad_len,))])
        labels = np.hstack([labels, 255 * np.ones((pad_len,))])

    return depth, refl, labels_image, labels, py, px, points_xyz, points_refl



def _transorm_test(depth, refl, labels_image, labels, py, px, points_xyz, points_refl, new_h=289, new_w=4097):
    depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    refl = cv2.resize(refl, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    labels_image = cv2.resize(labels_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    py = 2 * (py / 65.0 - 0.5)
    px = 2 * (px / 2049.0 - 0.5)

    return depth, refl, labels_image, labels, py, px, points_xyz, points_refl


class SemanticKitti(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path, split: str, new_h: int=289, new_w: int=4097) -> None:
        dataset_dir = Path(dataset_dir)
        self.split = split
        self.seqs = splits[split]
        self.dataset_dir = dataset_dir
        self.sweeps = []
        self.new_h = new_h
        self.new_w = new_w
        for seq in self.seqs:
            seq_str = f"{seq:0>2}"
            seq_path = dataset_dir / seq_str / "velodyne"
            for sweep in seq_path.iterdir():
                self.sweeps.append((seq_str, sweep.stem))

    def __getitem__(self, index):
        seq, sweep = self.sweeps[index]
        sweep_file = self.dataset_dir / seq / "velodyne" / f"{sweep}.bin"
        points = np.fromfile(sweep_file.as_posix(), dtype=np.float32)
        points = points.reshape((-1, 4))
        points_xyz = points[:, :3]
        if self.split != "test":
            labels_file = self.dataset_dir / seq / "labels" / f"{sweep}.label"
            labels = np.fromfile(labels_file.as_posix(), dtype=np.int32)
            labels = labels.reshape((-1))
            labels &= 0xFFFF
            labels = np.vectorize(learning_map.get)(labels)
        else:
            labels = np.zeros((points.shape[0],))

        points_refl = points[:, 3]
        depth_image, refl_image, labels_image, py, px = do_range_projection(points_xyz, points_refl, labels)

        if self.split == "train":
            depth_image, refl_image, labels_image, labels, py, px, points_xyz, points_refl = _transorm_train(
                depth_image, refl_image, labels_image, labels, py, px, points_xyz, points_refl, new_h=self.new_h, new_w=self.new_w
            )
        else:
            depth_image, refl_image, labels_image, labels, py, px, points_xyz, points_refl = _transorm_test(
                depth_image, refl_image, labels_image, labels, py, px, points_xyz, points_refl, new_h=self.new_h, new_w=self.new_w
            )
        #TODO(nik): fix later
        #tree = kdtree(points_xyz)
        #_, knns = tree.query(points_xyz, k=7)

        if points_xyz.shape[0] < px.shape[0]:
            pad_len = px.shape[0] - points_xyz.shape[0]
            points_xyz = np.vstack([points_xyz, np.zeros((pad_len, 3))])
            points_refl = np.concatenate([points_refl, np.zeros(pad_len)])
            #knns = np.vstack([knns, np.zeros((pad_len, 7))])

        # normalize values to be between -10 and 10
        depth_image = 25 * (depth_image - 0.4)
        refl_image = 20 * (refl_image - 0.5)
        points_refl = 20 * (points_refl - 0.5)
        image = np.stack([depth_image, refl_image]).astype(np.float32)

        px = px[np.newaxis, :]
        py = py[np.newaxis, :]
        labels = labels[np.newaxis, :]

        res = {
            "image": image,
            "labels_image": labels_image,
            "labels": labels,
            "px": px,
            "py": py,
            "points_xyz": points_xyz,
            "points_refl": points_refl,
            #"knns": knns,
        }

        if self.split in ["test", "val"]:
            res["seq"] = seq
            res["sweep"] = sweep

        return res['image'], res['labels_image'], -1, -1, -1 # dummy values

    def decode_segmap(self, img):
        """
        as rgb channel, height, width image
        """
        img_height, img_width = img.shape

        img_color = np.zeros((img_height, img_width, 3), np.uint8)
        for row in range(img_height):
            for col in range(img_width):
                label = img[row, col]
                label_inv = map_inv[label]
                color_bgr = color_map[label_inv]
                img_color[row, col] = np.array([color_bgr[2], color_bgr[1], color_bgr[0]])

        img_color = np.swapaxes(img_color, 2, 0)
        img_color = np.swapaxes(img_color, 1, 2)
        return img_color

    def __len__(self):
        return len(self.sweeps)


def label_image_to_kitti_colors(img):
    """
    as rgb channel, height, width image
    """
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3), np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            label_inv = map_inv[label]
            color_bgr = color_map[label_inv]
            img_color[row, col] = np.array([color_bgr[2], color_bgr[1], color_bgr[0]])

    img_color = np.swapaxes(img_color, 2, 0)
    img_color = np.swapaxes(img_color, 1, 2)
    return img_color


def label_img_to_color(img, cmap):
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3), np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(cmap[label])

    return img_color


def do_range_projection(
    points: np.ndarray, reflectivity: np.ndarray, labels: np.ndarray, W: int = 2049, H: int = 65,
):
    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, -scan_x)
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

    new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
    proj_y = np.zeros_like(proj_x)
    proj_y[new_raw] = 1
    proj_y = np.cumsum(proj_y)
    # scale to image size using angular resolution
    proj_x = proj_x * W - 0.001

    px = proj_x.copy()
    py = proj_y.copy()

    proj_x = np.floor(proj_x).astype(np.int32)
    proj_y = np.floor(proj_y).astype(np.int32)

    # order in decreasing depth
    order = np.argsort(depth)[::-1]

    depth = depth[order]
    reflectivity = reflectivity[order]
    labels = labels[order]

    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = np.zeros((H, W))
    proj_range[proj_y, proj_x] = 1.0 / depth

    proj_reflectivity = np.zeros((H, W))
    proj_reflectivity[proj_y, proj_x] = reflectivity

    proj_labels = np.zeros((H, W))
    proj_labels[proj_y, proj_x] = labels
    return proj_range, proj_reflectivity, proj_labels, py, px


learning_map = {
    0: 255,  # "unlabeled"
    1: 255,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: 255,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 8,  # "lane-marking" to "road" ---------------------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: 255,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 0,  # "moving-car" to "car" ------------------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 5,  # "moving-person" to "person" ------------------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 3,  # "moving-truck" to "truck" --------------------------------mapped
    259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mappe
}
class_names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]
map_inv = {
    0: 10,  # "car"
    1: 11,  # "bicycle"
    2: 15,  # "motorcycle"
    3: 18,  # "truck"
    4: 20,  # "other-vehicle"
    5: 30,  # "person"
    6: 31,  # "bicyclist"
    7: 32,  # "motorcyclist"
    8: 40,  # "road"
    9: 44,  # "parking"
    10: 48,  # "sidewalk"
    11: 49,  # "other-ground"
    12: 50,  # "building"
    13: 51,  # "fence"
    14: 70,  # "vegetation"
    15: 71,  # "trunk"
    16: 72,  # "terrain"
    17: 80,  # "pole"
    18: 81,  # "traffic-sign
    255: 0,
}

color_map = {  # bgr
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0],
}

train_color_map = {i: color_map[j] for i, j in map_inv.items()}
