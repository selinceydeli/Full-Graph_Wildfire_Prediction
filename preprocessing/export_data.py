import numpy as np
import geopandas as gpd
import pandas as pd
from typing import List


def export_distance_matrix(grid: gpd.geodataframe.GeoDataFrame):
    cent = grid.geometry.centroid
    N = len(cent)
    A = np.zeros(shape=(N, N))

    grid_points = grid["geometry"].to_numpy()
    P1s = np.array(list(map(lambda x: np.array([x.centroid.x, x.centroid.y]), grid_points[:])))
    P2s = np.array(list(map(lambda x: np.array([x.centroid.x, x.centroid.y]), grid_points[:])))

    A = np.sum((P1s[:, None, :] - P2s[None, :, :]) ** 2, axis=2) ** 0.5
    np.fill_diagonal(A, 0.0)  # ensure exact zero self-distances
    filename = "data/distance_matrix.npy"
    np.save(filename, A)

    # Sanity check to make sure the dist is in meters, not latitude
    assert grid.crs and not grid.crs.is_geographic

    print("Saved distance matrix as", filename)


def construct_timeseries_data(graphs: List[gpd.geodataframe.GeoDataFrame]):
    long = pd.concat(
        [
            g.copy()  # ensure consistent dtype
            for g in graphs
        ],
        axis=0,
    )

    print(long.columns)

    N = len(long.node_id.unique())
    T = len(long.DAY.unique())
    print(long.DAY.unique())
    M = len(long.columns)

    columns = list(long.columns)
    columns.remove("DAY")
    columns.remove("node_id")

    print("N x M x T:", f'{N} x {M} x {T}')

    print("rows", len(long), "expected", N * T, "ratio", len(long) / (N * T))
    dup_counts = (long.groupby(["node_id", "DAY"]).size())
    print("num duplicated (node_id,DAY) pairs:", (dup_counts > 1).sum())
    timeseries_data = long.sort_values(by=['node_id', 'DAY'])[columns].values.reshape(N, T, M - 2)
    filename = "data/timeseries_data.npy"
    np.save(filename, timeseries_data)

    print("Saved distance matrix as", filename)
