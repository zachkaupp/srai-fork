"""Tests for grid utils."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from srai.constants import WGS84_CRS
from srai.embedders.s2vec.grid_utils import get_grid_patches

CRS = "EPSG:32633"
CELL_SIZE = 40
IMG_SIZE = 2


def test_get_grid_patches() -> None:
    """Test that images cover the area without overlap and patches are ordered row by row."""
    # spans image columns 0-1 in image row 0 (y from -80 to 0)
    area_gdf = gpd.GeoDataFrame(geometry=[box(50, -70, 130, -10)], crs=CRS)
    patches_gdf, joint_gdf = get_grid_patches(
        area_gdf=area_gdf, crs=CRS, origin=(0, 0), cell_size=CELL_SIZE, img_size=IMG_SIZE
    )

    img_ids = list(joint_gdf.index.get_level_values("img_id").unique())
    assert img_ids == sorted(img_ids)
    assert len(img_ids) == 2
    assert len(patches_gdf) == 2 * IMG_SIZE**2
    assert patches_gdf.index.is_unique
    assert patches_gdf.crs == WGS84_CRS

    assert list(joint_gdf.loc[img_ids[0]].index) == ["0_0", "40_0", "0_-40", "40_-40"]
    assert list(joint_gdf.loc[img_ids[1]].index) == ["80_0", "120_0", "80_-40", "120_-40"]

    bounds = patches_gdf.to_crs(CRS).loc["40_-40"].geometry.bounds
    assert bounds == pytest.approx((40, -80, 80, -40), abs=1e-3)
