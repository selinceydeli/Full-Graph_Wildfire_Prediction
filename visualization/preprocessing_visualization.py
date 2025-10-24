import matplotlib.pyplot as plt
import os 
import geopandas as gpd 

def plot_grid_to_be_used(grid, data_plots_directory, img_name, title):

    _, ax = plt.subplots(figsize=(8, 8))
    grid.plot(ax=ax, edgecolor='black', facecolor='lightblue')
    ax.ticklabel_format(style='sci', scilimits=(4,4))
    ax.set_title(title)
    ax.set_axis_off()
    plt.savefig(os.path.join(data_plots_directory, img_name), dpi=300, bbox_inches="tight")
    plt.show()
    
def show_clusters(grid, coord_ref_sys, cluster_grid_centers, data_plots_directory, img_name):
    _, ax = plt.subplots(figsize=(8, 8))
    grid.to_crs(coord_ref_sys)

    center_points = [centroid for node_id, centroid in cluster_grid_centers.values()]

    cluster_points = gpd.GeoDataFrame(
        geometry = center_points,
        crs = coord_ref_sys
    )
    
    grid.plot(column="cluster_id", categorical=True, ax=ax, cmap="tab20")
    cluster_points.plot(ax=ax, color="black",  alpha=0.8, marker="s")
    ax.set_title("KMeans Clusters of Grid Squares")
    ax.set_axis_off()
    plt.savefig(os.path.join(data_plots_directory, img_name), dpi=300, bbox_inches="tight")
    plt.show()