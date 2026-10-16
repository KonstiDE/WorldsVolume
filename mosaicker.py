import os
import glob
import sys

import rioxarray as rxr
import xarray as xr
import numpy as np
from rioxarray.merge import merge_arrays


def mosaic(input_folder, site):
    output_file = f"/media/caipi/Toshi/volume/sites/dl/{site}/{site}.tif"

    if not os.path.exists(output_file):
        tif_files = glob.glob(os.path.join(input_folder, '*.tif'))

        print(f"Found {len(tif_files)} tiffs for id of '{site}'.", flush=True)

        assert len(tif_files) > 1

        datasets = [rxr.open_rasterio(fp, chunks=True) for fp in tif_files]
        mosaic = merge_arrays(datasets)

        mosaic.rio.to_raster(
            output_file,
            compress="lzw",
            BIGTIFF="YES"
        )

    else:
        print(f"{output_file} already exists. Skipping that.", flush=True)


if __name__ == '__main__':
    site = "phenix3"

    mosaic(f"/media/caipi/Toshi/volume/sites/raw/usa/{site}", site)