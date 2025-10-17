from typing import Tuple
from sklearn.cluster import KMeans
import geopandas as gpd
import numpy as np 
from shapely.geometry import Point as SPoint

def give_cluster_grid_centers(n_clusters: int, grid: gpd.geodataframe.GeoDataFrame, coord_ref_sys:str) -> Tuple[gpd.geodataframe.GeoDataFrame, gpd.geodataframe.GeoDataFrame]:

    grid.to_crs(coord_ref_sys)
    coords = np.array([(geom.y, geom.x) for geom in grid.centroid_grid])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init = 10)
    grid["cluster_id"] = kmeans.fit_predict(coords)

    cluster_points = [SPoint(xy[1], xy[0]) for xy in kmeans.cluster_centers_]
    # dict of the closest node to each cluster center
    cluster_grid_centers = {}

    for cluster_id, center in enumerate(cluster_points):
        cluster_nodes = grid[grid["cluster_id"] == cluster_id]
        # distances between the center of the cluster and all cluster nodes
        distances = cluster_nodes["centroid_grid"].distance(center)

        closest_node = distances.idxmin()
        closest_node_id = grid.loc[closest_node]["node_id"]
        closest_node_centroid = grid.loc[closest_node]["centroid_grid"]

        cluster_grid_centers[cluster_id] = (closest_node_id, closest_node_centroid)

    return cluster_grid_centers, grid

def get_node_cluster_centroid(point, node_cluster, cluster_grid_centers):
    # Given a node's centroid, returns the cluster it is part of
    # Return None if not found
    print(f"Point {point}")

    cluster_id = node_cluster.get(point)
    if cluster_id is None:
        print(f"No cluster id")
        return None

    node_id, centroid = cluster_grid_centers.get(cluster_id, (None, None))
    return centroid