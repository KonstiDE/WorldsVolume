import os
import uuid

import numpy as np
import rasterio
import rasterio as rio

import config.config as cfg

from tqdm import tqdm


def export_to_tif(frames, site_name):
    for df in tqdm(frames):
        data = df[0]
        dsm = df[1]
        transform = df[2]
        crs = df[3]

        full = np.vstack((dsm, data))

        profile = {
            "transform": transform,
            "width": 256,
            "height": 256,
            "crs": crs,
            "dtype": rasterio.float32,
            "count": 21,
            "compress": 'lzw'
        }

        with rio.open(os.path.join(os.path.join(
                cfg.get_frame_path(),
                f"frame_{site_name}_{uuid.uuid4()}.tif"
            )), "w", **profile) as dst:

            for channel in range(full.shape[0]):
                dst.write(full[channel, :, :], channel + 1)
