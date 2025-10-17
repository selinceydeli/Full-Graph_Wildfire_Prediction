from typing import List
import fiona
import geopandas as gpd
import pandas as pd
from shapely.strtree import STRtree


def add_water_to_grid(grid: gpd.geodataframe.GeoDataFrame, coord_ref_sys:str, rivers_data_path:str, layer_indices_for_dataset:List[int]) -> gpd.geodataframe.GeoDataFrame:
    # Read and concatenate all water layers
    water_gdfs = []
    for idx in layer_indices_for_dataset:
        layer_name = fiona.listlayers(rivers_data_path)[idx]
        gdf_layer = gpd.read_file(rivers_data_path, layer=layer_name)
        water_gdfs.append(gdf_layer)

    # Combine all layers into a single GeoDataFrame
    water_gdf = gpd.GeoDataFrame(pd.concat(water_gdfs, ignore_index=True))
    print("Combined water bodies:", water_gdf.shape)

    grid = grid.set_geometry('centroid_grid')

    # Ensure the CRS matches the water layer
    grid = grid.to_crs(water_gdf.crs)

    water_gdf = water_gdf[water_gdf.geometry.notnull()]
    water_geoms = water_gdf.geometry.values
    water_tree = STRtree(water_gdf.geometry.values)

    def nearest_water(point):
        if point is None or point.is_empty:
            return None
        # nearest() now returns index
        nearest_idx = water_tree.nearest(point)
        nearest_geom = water_geoms[nearest_idx]  # get the actual geometry
        return point.distance(nearest_geom)

    grid['dist_to_water'] = grid.geometry.apply(nearest_water)
    grid.to_crs(coord_ref_sys, inplace=True)
    grid = grid.set_geometry('geometry')
    grid.to_crs(coord_ref_sys, inplace=True)
    return grid 

def extract_fire_centroids(fire_data: gpd.geodataframe.GeoDataFrame) -> gpd.geodataframe.GeoDataFrame:
    fire_data["centroid_fire"] = fire_data['geometry'].apply(lambda y: y.centroid)
    return fire_data

def extract_fire_day(fire_data: gpd.geodataframe.GeoDataFrame) -> gpd.geodataframe.GeoDataFrame:
    fire_data["DAY"] = pd.to_datetime(fire_data["FIREDATE"], format='mixed').dt.date
    return fire_data