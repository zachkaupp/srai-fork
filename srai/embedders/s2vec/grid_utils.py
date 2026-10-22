"""Utility functions for square grids in a projected CRS."""

import math
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from srai.constants import REGIONS_INDEX, WGS84_CRS


def _format_coordinate(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def get_grid_patches(
    area_gdf: gpd.GeoDataFrame,
    crs: Union[str, int],
    origin: tuple[float, float],
    cell_size: float,
    img_size: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Cover an area with non-overlapping square images made of square patches.

    The grid is defined in a projected CRS. Patch corners lie on the lattice
    `origin + k * cell_size` and image corners on `origin + k * img_size * cell_size`.
    Every image that overlaps the area is returned with all of its patches.

    Parameters:
        area_gdf (gpd.GeoDataFrame): Area to cover.
        crs (Union[str, int]): Projected CRS the grid is defined in.
        origin (tuple[float, float]): (x, y) of any lattice point, in `crs` units.
        cell_size (float): Patch side length, in `crs` units.
        img_size (int): Number of patches along each side of an image.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Patches in WGS84, indexed by the `crs`
        coordinates of their top-left corner ("<x>_<y>"), and the joint gdf between images and
        patches. Image ids sort in the order the images are generated, and each image's
        patches are ordered row by row from the top-left.
    """
    area = area_gdf.to_crs(crs).union_all()
    img_extent = img_size * cell_size
    origin_x, origin_y = origin
    minx, miny, maxx, maxy = area.bounds

    # image rows count downwards from the origin, columns rightwards
    first_row = math.floor((origin_y - maxy) / img_extent)
    last_row = math.ceil((origin_y - miny) / img_extent)
    first_col = math.floor((minx - origin_x) / img_extent)
    last_col = math.ceil((maxx - origin_x) / img_extent)

    patch_ids, patch_geoms, img_patch_pairs = [], [], []
    img_count = 0
    for img_row in range(first_row, last_row):
        for img_col in range(first_col, last_col):
            left = origin_x + img_col * img_extent
            top = origin_y - img_row * img_extent
            img_geom = box(left, top - img_extent, left + img_extent, top)
            if not img_geom.intersects(area) or img_geom.touches(area):
                continue

            img_id = f"{img_count:08d}"
            img_count += 1
            for row in range(img_size):
                for col in range(img_size):
                    x = origin_x + (img_col * img_size + col) * cell_size
                    y = origin_y - (img_row * img_size + row) * cell_size
                    patch_id = f"{_format_coordinate(x)}_{_format_coordinate(y)}"
                    patch_ids.append(patch_id)
                    patch_geoms.append(box(x, y - cell_size, x + cell_size, y))
                    img_patch_pairs.append((img_id, patch_id))

    patches_gdf = gpd.GeoDataFrame(
        index=pd.Index(patch_ids, name=REGIONS_INDEX), geometry=patch_geoms, crs=crs
    ).to_crs(WGS84_CRS)
    img_patch_joint_gdf = gpd.GeoDataFrame(
        index=pd.MultiIndex.from_tuples(img_patch_pairs, names=["img_id", "patch_id"])
    )

    return patches_gdf, img_patch_joint_gdf
