from typing import List

from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely.strtree import STRtree
from meteostat import Point, Daily
from datetime import timedelta
from tqdm import tqdm
from pyproj import Transformer
import geopandas as gpd
import pandas as pd
import numpy as np
import fiona
import os
import rasterio
import warnings


def add_water_to_grid(grid: gpd.geodataframe.GeoDataFrame, coord_ref_sys: str, rivers_data_path: str,
                      layer_indices_for_dataset: List[int]) -> gpd.geodataframe.GeoDataFrame:
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


def enrich_with_intensity_data(graph, wildfire_severity_dir, date):
    graph["fire_intensity"] = 0
    year = str(date.year)
    graph_crs = graph.crs
    for filename in os.listdir(wildfire_severity_dir):
        if filename.endswith(".tiff") and year in filename:
            filepath = os.path.join(wildfire_severity_dir, filename)
            with rasterio.open(filepath) as src:
                # Ensure CRS match
                if graph.crs != src.crs:
                    graph = graph.to_crs(src.crs)
                    coords = [(row.geometry.centroid.x, row.geometry.centroid.y) if row["has_fire"] == 1 else None for
                              (index, row) in graph.iterrows()]
                values = np.array([src.sample(coord) if coord is not None else 0 for coord in coords])
                graph["fire_intensity"] = values
    graph = graph.to_crs(graph_crs)
    return graph


def enrich_with_air_data(graph: gpd.geodataframe.GeoDataFrame, cluster_grid_centers, day) -> GeoDataFrame:
    transformer = Transformer.from_crs(graph.crs, "epsg:4326", always_xy=True)
    cluster_weather_df = pd.DataFrame(
        columns=["cluster_id", "point", 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres',
                 'tsun'])
    warnings.filterwarnings('ignore', category=FutureWarning)
    for i in cluster_grid_centers.keys():
        (_, cluster_center) = cluster_grid_centers[i]
        lon, lat = transformer.transform(cluster_center.x, cluster_center.y)
        location = Point(lat, lon)
        location.radius = 150000
        today = pd.Timestamp(day)
        cluster_weather = Daily(location, today, today, model=True).fetch()
        cluster_weather["cluster_id"] = i
        cluster_weather["point"] = cluster_center
        cluster_weather_df = pd.concat([cluster_weather_df, cluster_weather], ignore_index=True)

    graph = graph.join(cluster_weather_df, on="cluster_id", lsuffix="right")

    return graph


def drop_unnecessary_columns(graph: gpd.geodataframe.GeoDataFrame) -> gpd.geodataframe.GeoDataFrame:
    graph = graph.drop(columns=["id", "index_right", "FIREDATE", "LASTUPDATE",
                                "COUNTRY", "COMMUNE", "AREA_HA", "geometry", "centroid_grid", "cluster_idright",
                                "snow", "wdir", "tsun", "wpgt", "cluster_id", "centroid_fire", "point", "index_right",
                                "SCLEROPH"])
    return graph


def interpolate_and_categorize_columns(graph: gpd.geodataframe.GeoDataFrame) -> DataFrame:
    graph["tavg"] = graph["tavg"].interpolate()
    graph["tmin"] = graph["tmin"].interpolate()
    graph["tmax"] = graph["tmax"].interpolate()
    graph["DAY"] = graph["DAY"].interpolate()
    graph["wspd"] = graph["wspd"].fillna(0)
    graph["pres"] = graph["wspd"].fillna(0)
    graph = pd.concat([graph, pd.get_dummies(graph["CLASS"])], axis=1).drop(["CLASS"], axis=1)
    to_be_filled = ["BROADLEA", "CONIFER", "MIXED", "OTHERNATLC", "TRANSIT", "AGRIAREAS", "ARTIFSURF", "OTHERLC",
                    "PERCNA2K"]
    for col in to_be_filled:
        graph[col] = graph[col].fillna(0)
    return graph


def split_graphs_into_days(grid: gpd.geodataframe.GeoDataFrame, fire_data: gpd.geodataframe.GeoDataFrame,
                           fires_by_day: gpd.geodataframe.GeoDataFrame, wildfire_severity_dir: str,
                           cluster_grid_centers: gpd.geodataframe.GeoDataFrame):
    fire_data = fire_data.to_crs(grid.crs)

    graphs = []
    for day in tqdm(fires_by_day):
        fire_data_day = fire_data[fire_data["DAY"] == day].copy()
        graph = gpd.sjoin(grid, fire_data_day, how="left", predicate="overlaps")
        graph = graph.drop_duplicates(subset="node_id", keep="first")

        graph["has_fire"] = graph["index_right"].notna().astype(int)
        graph = enrich_with_intensity_data(graph, wildfire_severity_dir, day)
        graph = enrich_with_air_data(graph, cluster_grid_centers, day)
        graph = drop_unnecessary_columns(graph)
        graph = interpolate_and_categorize_columns(graph)

        graphs.append(graph)

    return graphs
