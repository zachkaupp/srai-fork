"""S2Vec Dataset tests."""

import geopandas as gpd
import pandas as pd
import pytest
import torch
from shapely.geometry import box

from srai.embedders.s2vec.dataset import S2VecDataset
from srai.embedders.s2vec.grid_utils import get_grid_patches

CRS = "EPSG:32633"
CELL_SIZE = 40
IMG_SIZE = 4
NUM_IMAGES = 2


@pytest.fixture  # type: ignore
def empty_gdf() -> gpd.GeoDataFrame:
    """Get empty GeoDataFrame."""
    return gpd.GeoDataFrame(geometry=[])


@pytest.fixture  # type: ignore
def data_and_joint_dfs() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Get example regions for testing."""
    img_extent = CELL_SIZE * IMG_SIZE
    area_gdf = gpd.GeoDataFrame(geometry=[box(0, 0, NUM_IMAGES * img_extent, img_extent)], crs=CRS)
    patches_gdf, img_patch_joint_gdf = get_grid_patches(
        area_gdf=area_gdf, crs=CRS, origin=(0, 0), cell_size=CELL_SIZE, img_size=IMG_SIZE
    )
    data_df = pd.DataFrame(list(range(len(patches_gdf))), index=patches_gdf.index, columns=["data"])
    return data_df, img_patch_joint_gdf


def test_dataset_length(data_and_joint_dfs: tuple[pd.DataFrame, gpd.GeoDataFrame]) -> None:
    """Test if S2VecDataset constructs lookup tables correctly."""
    data_df, img_patch_joint_gdf = data_and_joint_dfs
    dataset = S2VecDataset(data_df, img_patch_joint_gdf)  # type: ignore
    assert len(dataset) == NUM_IMAGES


def test_dataset_get_item_shape(data_and_joint_dfs: tuple[pd.DataFrame, gpd.GeoDataFrame]) -> None:
    """Test if S2VecDataset items have the correct size."""
    data_df, img_patch_joint_gdf = data_and_joint_dfs
    dataset = S2VecDataset(data_df, img_patch_joint_gdf)  # type: ignore

    item_0 = dataset[0]

    assert isinstance(item_0, torch.Tensor)
    assert item_0.shape == torch.Size([IMG_SIZE**2, 1])
