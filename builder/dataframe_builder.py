import os
import rioxarray

import config.config as cfg
import rasterio as rio

from builder.tools import slice_n_dice, check_integrity_and_repair
from builder.export import export_to_tif

import xarray as xr


netdcf_vars = ["red"]


def walk(parts):
    for part in parts:
        print("==================================================")
        print("Beginning on part {}".format(part))

        dsm_path = os.path.join(cfg.get_raw_data_path(), f"{part}", "dsm_normalized_masked_res1m.tif")
        data_path = os.path.join(cfg.get_raw_data_path(), f"{part}", "mean_timeseries_sr.nc")

        dsm = rioxarray.open_rasterio(dsm_path)
        ds = xr.open_dataset(data_path)

        ds_resampled = ds.rio.reproject_match(dsm, resampling=rio.enums.Resampling.nearest)


        print("Slicing...")
        frames = slice_n_dice(ds_resampled, dsm, cfg.blob_size())


        print("Checking integrity...")
        frames = check_integrity_and_repair(frames)

        print("Exporting...")
        export_to_tif(frames=frames, site_name=part)
        print("==================================================")
