# graph_utils.py
from typing import Optional, Tuple

import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import random
import shapely.geometry
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import split
import math
import re
from tqdm import tqdm
import itertools
from collections import defaultdict
import adding_util


def load_graph(place: Optional[str] = None,
               filepath: Optional[str] = None,
               network_type: str = "drive", folder: str = "graphs", add_buildings = True, add_charging = True, default_speed = 30, max_buildings = 500):
    """
    Load a drivable graph. If `filepath` ends with .graphml, it will be loaded
    from disk; otherwise, download by `place` name.

    Adds edge attributes:
      - length (meters) if missing
      - speed_kph (estimated if missing)
      - travel_time (seconds)
    """
  
    if filepath:
      G = ox.load_graphml(filepath)
      
    elif place:
      os.makedirs(folder, exist_ok=True)
      
      # Create safe filename
      safe_place = place.replace(",", "").replace(" ", "_")
      filepath = os.path.join(folder, f"{safe_place}_graph.graphml")
    
      if os.path.exists(filepath):
        print(f"[Graph] Loading cached graph: {filepath}")
        G = ox.load_graphml(filepath)      
      else:
        print(f"[Graph] Downloading graph for {place}...")
        G = ox.graph_from_place(place, network_type=network_type)
        G = fix_missing_edge_attributes(G, default_speed)
        if (add_buildings):
          G, summary = adding_util.add_buildings_batch(G, place, max_snap_dist=1000, max_buildings=max_buildings, show_progress=True, walking_speed_kph = 5.0)
          print(summary)
        if (add_charging):
          G, chargingStations = add_charging_stations(G, place)
        G = fix_missing_edge_attributes(G, default_speed)
        G = ox.project_graph(G)
        # Force all street_count attributes to int type before saving
        nx.set_node_attributes(
          G, {n: int(float(d["street_count"])) for n, d in G.nodes(data=True) if "street_count" in d}, "street_count"
        )
        G = ox.distance.add_edge_lengths(G)      # adds 'length' in meters
        G = ox.routing.add_edge_speeds(G)             # adds 'speed_kph'
        fix_missing_edge_attributes(G, default_speed)
        G = ox.routing.add_edge_travel_times(G)       # adds 'travel_time' in seconds
        ox.save_graphml(G, filepath)
        print(f"[Graph] Saved graph to {filepath}")
    else:
        raise ValueError("Provide either `place` or `filepath` to load_graph().")
    return G


def nearest_node(G, x: float, y: float) -> int:
    """Return node id nearest to (x, y). x,y are lon,lat for typical OSMnx graphs."""
    return ox.distance.nearest_nodes(G, X=x, Y=y)


def route_metrics(G, u: int, v: int) -> Tuple[float, float]:
    """
    Return (travel_time_seconds, distance_meters) along the fastest path from u to v,
    using 'travel_time' as the edge weight.
    """
    if u == v:
        return 0.0, 0.0
    try:
        t_sec = nx.shortest_path_length(G, u, v, weight="travel_time")
        path = nx.shortest_path(G, u, v, weight="travel_time")
        dist_m = 0.0
        for a, b in zip(path[:-1], path[1:]):
            data = min(G[a][b].values(), key=lambda d: d.get("travel_time", 0.0))
            dist_m += float(data.get("length", 0.0))
        return float(t_sec), float(dist_m)
    except nx.NetworkXNoPath:
        return 1e9, 1e9

def get_randomNodes(G, seed: int):
  buildings = [n for n, d in G.nodes(data=True) if d.get("building")]
  random.seed(seed)
  random.r

def show_graph(G, highlight_buildings=True, highlight_charging=True):
  
    fig, ax = ox.plot_graph(
        G,
        node_size=5,
        node_color="lightgray",
        edge_color="gray",
        show=False,
        close=False
    )
  
    if highlight_buildings:
        building_nodes = [
          (d["x"], d["y"]) 
          for n, d in G.nodes(data=True) 
          if str(d.get("building")).lower() in ("true", "1") or d.get("type") == "building"
        ]
        buildings = [n for n, d in G.nodes(data=True) if "building" in d]
        print(len(building_nodes))
        print("Num building nodes in show:", len(buildings))
        if building_nodes:
            bx, by = zip(*building_nodes)
            ax.scatter(bx, by, c="red", s=8, label="Buildings", alpha=0.6)
          
    if highlight_charging:
        charging_nodes = [(d["x"], d["y"]) for n, d in G.nodes(data=True)
                          if d.get("type") == "charging_station"]
        if charging_nodes:
            cx, cy = zip(*charging_nodes)
            ax.scatter(cx, cy, c="blue", s=20, marker="s", label="Charging", alpha=0.8)
          
    ax.legend()
    plt.show()
  

def add_buildings_as_nodes(G, place, max_dist=100, max_buildings = 100):
  print("[Graph] Fetching buildings...")
  buildings = ox.features_from_place(place, tags={"building": True})
  
  if buildings.empty:
    print("[Graph] No buildings found.")
    return G
  if len(buildings) > maxbuildings:
    print(f"[Graph] Many buildings found, cutting to first {max_buildings} buildings for performance")
    buildings = buildings.head(max_buildings)
  crs = G.graph.get("crs")
  if crs is None:
    # Assume WGS84 if no CRS is set
    G.graph["crs"] = "EPSG:4326"
    crs = G.graph["crs"]
    
  # Ensure CRS is a pyproj.CRS object
  if isinstance(crs, str):
    crs = pyproj.CRS.from_string(crs)
    
  if crs.is_geographic:
    G = ox.project_graph(G, to_crs="EPSG:3857")
  
  if buildings.crs.is_geographic:
    buildings = buildings.to_crs(G.graph["crs"])
    
  centroids = buildings.centroid
  print(f"[Graph] Found {len(centroids)} buildings.")
  
  start_id = max(G.nodes) + 1 if all(isinstance(n, int) for n in G.nodes) else len(G.nodes) + 1
  new_node_id = start_id
  added=0
  for i, pt in enumerate(tqdm(centroids, desc="[Graph] Adding building nodes")):
    # create building node
    building_node = new_node_id
    new_node_id += 1
    G.add_node(building_node, x=float(pt.x), y=float(pt.y),
          building=True, type="building")

    # find nearest edge (road segment)
    try:
      u, v, key = ox.distance.nearest_edges(G, X=pt.x, Y=pt.y, return_dist=False)
      edge_data = G.get_edge_data(u, v, key)
      
      # create a "connection node" projected on this edge
      line: LineString = edge_data.get("geometry", LineString([
        (G.nodes[u]["x"], G.nodes[u]["y"]),
        (G.nodes[v]["x"], G.nodes[v]["y"])
      ]))
      proj_point = line.interpolate(line.project(pt))
      
      connection_node = new_node_id
      new_node_id += 1
      G.add_node(connection_node, x=float(proj_point.x), y=float(proj_point.y),
            connection=True, type="connection")
      
      # connect building to connection node (short walk edge)
      dist = pt.distance(proj_point)
      if dist <= max_dist:
        G.add_edge(building_node, connection_node,
              length=dist, type="building_link", speed_kph=5)
        G.add_edge(connection_node, building_node,
              length=dist, type="building_link", speed_kph=5)
        
        # optionally: connect connection node into road network
        dist_to_u = Point(G.nodes[u]["x"], G.nodes[u]["y"]).distance(proj_point)
        dist_to_v = Point(G.nodes[v]["x"], G.nodes[v]["y"]).distance(proj_point)
        
        # remove original edge u-v and replace with u-connection-v split
        split_edge_at_point(G, u, v, key, proj_point, connection_node, edge_data)
        # keep direction if one-way
        if not edge_data.get("oneway", False):
          G.add_edge(v, connection_node,
                length=dist_to_v, type="road_split",
                speed_kph=edge_data.get("speed_kph", 30))
          G.add_edge(connection_node, u,
                length=dist_to_u, type="road_split",
                speed_kph=edge_data.get("speed_kph", 30))
          
        added += 1
        
    except Exception as e:
      continue
    
  print(f"[Graph] Added {added} buildings with road connections.")
  clean_graph_nodes(G)
  return G


def add_charging_stations(G, place: str):
  """
  Load EV charging stations from OSM and add them as special nodes to the graph.
  They are snapped to the nearest road node.

  Parameters
  ----------
  G : networkx.MultiDiGraph
    Road graph
  place : str
    Place name (e.g., "Detmold, Germany")

  Returns
  -------
  G : networkx.MultiDiGraph
    Graph with charging stations tagged
  stations : list
    List of charging station node IDs
  """
  # Query charging stations
  chargers = ox.features_from_place(place, tags={"amenity": "charging_station"})
  
  stations = []
  for _, row in chargers.iterrows():
    if "geometry" not in row:
      continue
    geom = row.geometry
    if geom.is_empty:
      continue
    
    # Some are Points, others could be small polygons → use centroid
    pt = geom.centroid
    nearest_node = ox.distance.nearest_nodes(G, pt.x, pt.y)
    
    # Tag this node as a charging station
    G.nodes[nearest_node]["charging_station"] = True
    stations.append(nearest_node)
    
  print(f"[Graph] Added {len(stations)} charging stations.")
  return G, stations

def add_charging_stations_by_coords(G, coords):
  stations = []
  for lat, lon in coords:
    node = ox.distance.nearest_nodes(G, lon, lat)  # note lon/lat order!
    G.nodes[node]["charging_station"] = True
    stations.append(node)
  return stations



def add_random_charging_stations(G, n_stations=5, seed=42):
  random.seed(seed)
  nodes = list(G.nodes())
  stations = random.sample(nodes, n_stations)
  for node in stations:
    G.nodes[node]["charging_station"] = True
  return stations


def check_bad_edges(G):
  bad_edges = []
  for u, v, k, data in G.edges(keys=True, data=True):
    if (
      data.get("length") is None 
      or (isinstance(data.get("length"), float) and math.isnan(data["length"]))
      or data.get("speed_kph") is None
      or (isinstance(data.get("speed_kph"), float) and math.isnan(data["speed_kph"]))
    ):
      bad_edges.append((u, v, k, data))
  print(f"Found {len(bad_edges)} edges with missing/NaN attributes.")
  return bad_edges

def parse_maxspeed(value):
  """Try to extract a numeric speed (kph) from messy OSM maxspeed tags."""
  if value is None:
    return None
  if isinstance(value, (int, float)) and not math.isnan(value):
    return int(value)
  if isinstance(value, list):
    speeds = [parse_maxspeed(v) for v in value]
    speeds = [s for s in speeds if s is not None]
    return min(speeds) if speeds else None
  if isinstance(value, str):
    # Extract first number
    match = re.search(r"\d+", value)
    if match:
      return int(match.group())
  return None


def fix_missing_edge_attributes(G, default_speed=30):
  fixed_len = fixed_spd = 0
  for u, v, k, data in G.edges(keys=True, data=True):
    # Fix length
    if (
      data.get("length") is None 
      or (isinstance(data.get("length"), float) and math.isnan(data["length"]))
    ):
      if "geometry" in data:
        data["length"] = data["geometry"].length
      else:
        data["length"] = 1.0
      fixed_len += 1
    
    # Fix speed
    if (
      data.get("speed_kph") is None 
      or (isinstance(data.get("speed_kph"), float) and math.isnan(data["speed_kph"]))
    ):
      spd = parse_maxspeed(data.get("maxspeed"))
      data["speed_kph"] = spd if spd is not None else default_speed
      fixed_spd += 1
    
  print(f"[Graph] Fixed {fixed_len} edges with missing length, {fixed_spd} edges with missing speed.")
  return G

def clean_graph_nodes(G):
  bad_nodes = [n for n, d in G.nodes(data=True) if "x" not in d or "y" not in d]
  if bad_nodes:
    print(f"[Graph] Removing {len(bad_nodes)} nodes without coordinates.")
    G.remove_nodes_from(bad_nodes)
  return G

