import warnings

import numpy as np
from icechunk import xarray
from rasterio.windows import Window
from rioxarray import rioxarray

from tqdm import tqdm
from scipy import interpolate
from scipy.interpolate import griddata
import xarray as xr
import rasterio as rio

import config.config as cfg


def slice_n_dice(data: xr.Dataset, mask: xr.Dataset, t: int):
    rows, cols = data.rio.shape

    base_transform = data.rio.transform()
    crs = data.rio.crs

    frames = []

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            row_end = min(r + t, rows)
            col_end = min(c + t, cols)

            h = row_end - r
            w = col_end - c

            window = Window(col_off=c, row_off=r, width=w, height=h)

            window_transform = rio.windows.transform(
                window, base_transform
            )

            window_data = data.isel(y=slice(r, row_end), x=slice(c, col_end))
            window_mask = mask.isel(y=slice(r, row_end), x=slice(c, col_end))

            arr_data = (
                window_data.to_array().values[0]
                if isinstance(window_data, xr.Dataset)
                else window_data.values
            )
            arr_mask = (
                window_mask.to_array().values[0]
                if isinstance(window_mask, xr.Dataset)
                else window_mask.values
            )

            frames.append((arr_data, arr_mask, window_transform, crs))

    return frames


def check_integrity_and_repair(data_frames):
    valid_frames = []

    count = 0

    for df in tqdm(data_frames):
        data = df[0]
        dsm = df[1]
        transform = df[2]
        crs = df[3]

        if not np.all(dsm == np.nan):
            nan_percentage = np.sum(np.isnan(dsm))

            if nan_percentage < 1:
                interpolate_nan_bands_(dsm)
                valid_frames.append((data, dsm, transform, crs))
            else:
                count += 1

    if len(valid_frames) == 0:
        raise Exception("Variable valid_frame is empty after check_integrity() was called.")
    else:
        print(f"Omitted {count} data frames. Total data frames returning {len(valid_frames)} that were also repaired")

    return valid_frames


def check_array_integrity(array):
    return len(np.unique(array)) > 1


def raw_slicing_no_channels(data, t):
    rows, cols = data.shape
    frames = []

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            tile_data = data[r:r + t, c:c + t]
            if tile_data.shape != (t, t):
                padded_data = np.zeros((t, t))
                padded_data[:tile_data.shape[0], :tile_data.shape[1]] = tile_data

                frames.append(padded_data)
            else:
                frames.append(tile_data)

    return frames


def pad_image_rgb(img, color, border_width=.1):
    h, w, c = img.shape

    border = int(float(min(h, w) * border_width))
    result = np.full((h + 2 * border, w + 2 * border, c), color, dtype=img.dtype)

    result[border:-border, border:-border] = img

    return result


def fix_missing_vals(data_frames):
    for df in tqdm(data_frames):
        interpolate_nan_bands_(df.data_stack)


def interpolate_nan_bands_(array, method='linear'):
    for band_index in range(array.shape[2]):
        band = array[:, :, band_index]
        nan_mask = np.isnan(band)

        if not np.any(nan_mask) or np.all(nan_mask):
            continue

        valid_y, valid_x = np.nonzero(~nan_mask)
        valid_values = band[~nan_mask]

        grid_y, grid_x = np.mgrid[0:band.shape[0], 0:band.shape[1]]

        interpolated = griddata(
            points=(valid_y, valid_x),
            values=valid_values,
            xi=(grid_y, grid_x),
            method=method
        )

        # Overwrite only the NaN positions (retains boundaries if linear cannot extrapolate)
        band[nan_mask] = interpolated[nan_mask]

    return array


if __name__ == '__main__':
    a = np.random.randn(512, 512, 4)

    a[70, 342, 0] = np.nan
    print(a[70, 342, 0])

    a = interpolate_nan_bands_(a)
    print(a[70, 342, 0])