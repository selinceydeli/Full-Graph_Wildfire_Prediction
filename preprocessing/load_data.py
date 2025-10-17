from typing import Tuple
import geopandas as gpd
import numpy as np 
import shapely
from shapely.geometry import box

def load_fire_data(fire_data_path:str, coord_ref_sys:str, country:str = "PT") -> gpd.geodataframe.GeoDataFrame: 
    # GeoPandas looks in the same directory for data files by default (.dbf, .shx, etc.)
    fire_data = gpd.read_file(fire_data_path)
    fire_data = fire_data.to_crs(coord_ref_sys) # reproject to our fixed CRS
    fire_data = fire_data[fire_data["COUNTRY"] == country]
    
    print("Loaded fire dataset with", fire_data["FIREDATE"].nunique(), "unique fires")
    return fire_data

def extract_polygon(country_boundary_dataset_path:str, country:str, coord_ref_sys:str): 
    
    world = gpd.read_file(country_boundary_dataset_path)
    portugal = world[world["NAME"] == country]

    # https://epsg.io/25829
    # Reprojection to our fixed CRS 
    proj = portugal.to_crs(coord_ref_sys) # degrees -> meters
    mainland = proj.iloc[proj.geometry.area.argmax()]
    mainland_polygon = mainland.geometry

    # Extract bordering box around Portugal
    minx, miny, maxx, maxy = mainland_polygon.bounds
    print("Country bounds", f"minx: {minx}, miny: {miny}, maxx: {maxx}, maxy: {maxy}")
    return mainland, mainland_polygon, minx, miny, maxx, maxy 

def cut_size(minx:int, miny:int, maxx:int, maxy:int, decrease_bottom:int, decrease_top:int, 
             decrease_left:int, decrease_right) -> Tuple[int, int, int, int]:
    height = maxy - miny
    width = maxx - minx
    
    assert decrease_bottom + decrease_bottom <= 1.0
    assert decrease_left + decrease_right <= 1.0

    cut_minx = minx + width * decrease_left
    cut_miny = miny +  height * decrease_bottom
    cut_maxx = maxx - width * decrease_right
    cut_maxy = maxy - (height * decrease_top)
    
    print(f"New minx: {cut_minx}, new miny: {cut_maxy}, new maxx: {cut_maxx}, new maxy: {cut_maxy}")
    
    return cut_minx, cut_miny, cut_maxx, cut_maxy

def create_grid(grid_size:int, mainland_polygon: shapely.geometry.polygon.Polygon, 
                coord_ref_sys:str, minx:int, miny:int, maxx: int, maxy: int) -> gpd.geodataframe.GeoDataFrame:
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)
    grid_squares = []
    for x in x_coords:
        for y in y_coords:
            # this makes a full rectangle grid
            cell = box(x, y, x + grid_size, y + grid_size)
            # only add cells that are inside mainland portugal (leave out the sea)
            if mainland_polygon.contains(cell):
                grid_squares.append(cell)

    grid = gpd.GeoDataFrame({"geometry": grid_squares}, crs=coord_ref_sys)
    print("Successfully created grid!")
    print(f"Num nodes = {len(grid_squares)}")
    
    return grid
    