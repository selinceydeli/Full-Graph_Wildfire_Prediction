import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence


def export_distance_matrix(
        grid: gpd.GeoDataFrame,
        outpath: str = "data/distance_matrix.npy",
        nodes: Optional[Sequence[int]] = None,  # pass np.load("data/nodes.npy")
        assert_projected: bool = True,
) -> np.ndarray:
    if assert_projected:
        assert grid.crs is not None and not grid.crs.is_geographic, "Grid CRS must be projected (meters)."

    # Align grid rows to the nodes order used in timeseries data
    if nodes is not None:
        idxer = pd.Index(grid["node_id"]).get_indexer(nodes)
        if (idxer < 0).any():
            missing = np.asarray(nodes)[idxer < 0]
            raise ValueError(f"{len(missing)} node_ids in nodes.npy not in grid: {missing[:10]}...")
        grid = grid.iloc[idxer].copy()
        assert np.array_equal(grid["node_id"].to_numpy(), np.asarray(nodes)), "Grid not aligned to nodes order."

    # Compute centroid coordinates and pairwise distances
    centroids = grid.geometry.centroid
    P = np.column_stack([centroids.x.values, centroids.y.values]).astype(np.float32)
    diff = P[:, None, :] - P[None, :, :]
    A = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(A, 0.0)

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    np.save(outpath, A)
    print(
        f"Saved distance matrix as {outpath}, shape={A.shape}, "
        f"symmetric={np.allclose(A, A.T)}, zero_diag={np.allclose(np.diag(A), 0.0)}"
    )
    return A


def construct_timeseries_data(graphs: list, reconstruct=False, save_path="data/timeseries_data.npy", label_col="has_fire"):
    import json
    from pathlib import Path

    long = pd.concat([g.copy() for g in graphs], axis=0, ignore_index=True)
    long["DAY"] = pd.to_datetime(long["DAY"], errors="coerce").dt.normalize()
    long = long.dropna(subset=["node_id", "DAY"])
    long["node_id"] = long["node_id"].astype(int)

    long = long.drop(columns=[c for c in ["geometry", "centroid_grid", "FireSeason", "30DAYS", "7DAYS", "prcp"] if c in long.columns], errors="ignore")

    # Aggregate to get 1 row per (node_id, DAY)
    key = ["node_id", "DAY"]
    value_cols = [c for c in long.columns if c not in key]

    pre_counts = long.groupby(key, as_index=False).size()
    collisions = int((pre_counts["size"] > 1).sum())
    print(f"[diagnostics] rows={len(long)}, unique pairs={len(pre_counts)}, collisions={collisions}")

    # Collapse Duplicates
    collapse_cache = Path("data/collapsed_long.pkl")
    if collapse_cache.exists() and not reconstruct:
        long = pd.read_pickle(collapse_cache)
    else:
        value_cols = [c for c in long.columns if c not in key]
        sum_cols = {"prcp", "fire_intensity", "30DAYS", "7DAYS"}
        max_cols = {"has_fire", "FireSeason"}
        mean_cols = {"tavg", "tmin", "tmax", "wspd", "pres"}
        print("I got here")
        agg = {}
        for c in value_cols:
            if c in sum_cols:
                agg[c] = "sum"
            elif c in max_cols:
                agg[c] = "max"
            elif c in mean_cols:
                agg[c] = "mean"
            else:
                agg[c] = "first"
        print("I got here")
        long = long.groupby(key, as_index=False).agg(agg)
        print("I am pickling")
        long.to_pickle(collapse_cache)

    # Create the rectangular panel
    nodes = np.sort(long["node_id"].unique())
    days = np.sort(long["DAY"].unique())
    panel_index = pd.MultiIndex.from_product([nodes, days], names=key)
    panel = long.set_index(key).reindex(panel_index)
    
    print("I got here")
    static_candidates = {"PROVINCE", "BROADLEA", "CONIFER", "MIXED", "TRANSIT", "OTHERNATLC",
                         "AGRIAREAS", "ARTIFSURF", "OTHERLC", "PERCNA2K", "dist_to_water"}
    static_cols = [c for c in panel.columns if c in static_candidates]
    if static_cols:
        filled = panel.groupby(level=0)[static_cols].ffill().bfill()
    for c in static_cols:
        panel[c] = pd.to_numeric(filled[c], errors='ignore')
        print(panel[c])
    # Keep only the numeric columns
    num_panel = panel.select_dtypes(include=[np.number, "bool"]).copy()
    bool_cols = list(num_panel.select_dtypes(include=["bool"]).columns)
    if bool_cols:
        num_panel[bool_cols] = num_panel[bool_cols].astype(float)
    print("I got here 2")
    # Print dropped non-numeric columns
    dropped = [c for c in panel.columns if c not in num_panel.columns]
    if dropped:
        print(f"[info] Dropping non-numeric columns from tensor: {dropped}")

    N, T = len(nodes), len(days)
    assert num_panel.shape[0] == N * T, f"Rectangularity failed: {num_panel.shape[0]} != {N * T}"

    all_cols = list(num_panel.columns)
    if label_col not in all_cols:
        raise ValueError(f"{label_col} not in {all_cols}")
    feat_cols = [c for c in all_cols if c != label_col]

    X = num_panel[feat_cols].to_numpy(dtype=float).reshape(N, T, len(feat_cols))
    y = num_panel[label_col].to_numpy(dtype=float).reshape(N, T)

    Path("data").mkdir(parents=True, exist_ok=True)
    np.save(save_path, X)
    np.save("data/labels.npy", y)
    np.save("data/nodes.npy", nodes)
    np.save("data/days.npy", days)
    with open("data/feat_cols.json", "w") as f:
        json.dump(feat_cols, f)

    print(f"Saved timeseries to {save_path} with shape {X.shape} (N={N}, T={T}, M={len(feat_cols)})")
    return X, y, nodes, days, feat_cols
