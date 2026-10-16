import os.path

import numpy as np
from ray import tune

from model.unet.unet_model import UNET
from model.dense121_unet.dense121_unet_model import DenseNet121

config = {
    "base_path": "/home/caipi/PycharmProjects/WorldsVolume",
    "raw_data_path": "/media/caipi/Toshi/volume/sites/dl/",
    "frame_path": "/media/caipi/Toshi/volume/frames/",
    "split": {
        "train": (0.7, "train/"),
        "validation": (0.2, "validation/"),
        "test": (0.1, "test/")
    },
    "num_workers": 0,
    "pin_memory": True,
    "blob_size": 256,
    "ray_config": {
        "lr": tune.grid_search([3e-04]),
        "batch_size": tune.grid_search([4])
    },
    "device": "cuda:0"
}


def get_frame_path():
    return config["frame_path"]


def get_train_path():
    return os.path.join(get_frame_path(), config["split"]["train"][1])


def get_validation_path(version):
    return os.path.join(get_frame_path(), config["split"]["validation"][1])


def get_test_path():
    return os.path.join(get_frame_path(), config["split"]["test"][1])


def batch_size():
    return config["batch_size"]


def num_workers():
    return config["num_workers"]


def pin_memory():
    return config["pin_memory"]


def blob_size():
    return config["blob_size"]


def get_base_path():
    return config["base_path"]


def get_raw_data_path():
    return config["raw_data_path"]


def get_r_home():
    return config["r_home"]


def get_r_x64_dll():
    return config["r_dll"]


def get_features():
    return config["features"]


def get_auxiliary_raster():
    return config["auxiliary_rasters"]


def get_split_map():
    return config["split"]


def get_no_data_value():
    return config["no_data_value"]


def get_ray_config():
    return config["ray_config"]


def get_device():
    return config["device"]

