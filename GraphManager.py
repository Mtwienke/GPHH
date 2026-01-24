#!/usr/bin/env python3

from typing import Optional, Tuple

import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import random
import shapely.geometry
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString, MultiPoint, GeometryCollection, MultiLineString 
from shapely.geometry.base import GeometrySequence
from shapely.ops import split
import math
import re
from tqdm import tqdm
import itertools
from collections import defaultdict
import adding_util
import utils
import pyproj
import heapq
from math import inf
import numpy as np

class GraphManager:
	def __init__(self, place: Optional[str] = None, 
		filepath: Optional[str] = None,
		network_type: str = "drive", graph_folder: str = "graphs", include_buildings = True, include_charging = True, default_speed = 30, max_buildings = 500, max_snap_dist=1000, logger: utils.Logger =None):
		self.place = place
		self.filepath = filepath
		self.logger = logger
		self.networkType = network_type
		self.graphFolder = graph_folder
		self.includeBuildings = include_buildings
		self.includeCharging = include_charging
		self.defaultSpeed = default_speed
		self.maxBuildings = max_buildings
		self.maxSnapDistance = max_snap_dist
		self.stations = []
		self.graph = None
		self.travelTimeCache = None
		self.travelDistanceCache = None


	def log(self, message):
		if self.logger == None:
			print(message)
		else:
			self.logger.log(message)
	
	def generateGraph(self):
		self.downloadBaseGraph()
		print (self.graph)
		if self.includeBuildings:
			self.log(self.add_buildings())
		if self.includeCharging:
			self.stations = self.add_charging_stations()
		self.fix_missing_edge_attributes(self.defaultSpeed)
		self.graph = ox.project_graph(self.graph)
		# Force all street_count attributes to int type before saving
		nx.set_node_attributes(
			self.graph, {n: int(float(d["street_count"])) for n, d in self.graph.nodes(data=True) if "street_count" in d}, "street_count"
		)
		self.graph = ox.distance.add_edge_lengths(self.graph)      # adds 'length' in meters
		self.graph = ox.routing.add_edge_speeds(self.graph)             # adds 'speed_kph'
		self.fix_missing_edge_attributes(self.defaultSpeed)
		self.graph = ox.routing.add_edge_travel_times(self.graph)       # adds 'travel_time' in seconds
		ox.save_graphml(self.graph, self.filepath)
		self.log(f"[Graph] Saved graph to {self.filepath}")
		
	def loadGraph(self, filepath=None):
		if filepath:
			G = ox.load_graphml(filepath)
			
		elif self.place:
			os.makedirs(self.graphFolder, exist_ok=True)
			
			# Create safe filename
			safe_place = self.place.replace(",", "").replace(" ", "_")
			filepath = os.path.join(self.graphFolder, f"{safe_place}_graph.graphml")
			
			if os.path.exists(filepath):
				print(f"[Graph] Loading cached graph: {filepath}")
				G = ox.load_graphml(filepath)      
			else:
				self.generateGraph()
		
			
	def downloadBaseGraph(self):
		self.log(f"[Graph] Downloading graph for {self.place}...")
		self.graph = ox.graph_from_place(self.place, network_type=self.networkType)
		self.fix_missing_edge_attributes(self.defaultSpeed)
		
	def show_graph(self, highlight_buildings=True, highlight_charging=True):
		fig, ax = ox.plot_graph(
				self.graph,
				node_size=5,
				node_color="lightgray",
				edge_color="gray",
				show=False,
				close=False
		)
	
		if highlight_buildings:
				building_nodes = [
					(d["x"], d["y"]) 
					for n, d in self.graph.nodes(data=True) 
					if str(d.get("building")).lower() in ("true", "1") or d.get("type") == "building"
				]
				buildings = [n for n, d in self.graph.nodes(data=True) if "building" in d]
				print(len(building_nodes))
				print("Num building nodes in show:", len(buildings))
				if building_nodes:
						bx, by = zip(*building_nodes)
						ax.scatter(bx, by, c="red", s=8, label="Buildings", alpha=0.6)
					
		if highlight_charging:
				charging_nodes = [(d["x"], d["y"]) for n, d in self.graph.nodes(data=True)
													if d.get("type") == "charging_station"]
				if charging_nodes:
						cx, cy = zip(*charging_nodes)
						ax.scatter(cx, cy, c="blue", s=20, marker="s", label="Charging", alpha=0.8)
					
		ax.legend()
		plt.show()
			
		
	def fix_missing_edge_attributes(self, default_speed=30):
		fixed_len = fixed_spd = 0
		for u, v, k, data in self.graph.edges(keys=True, data=True):
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
				spd = self.parse_maxspeed(data.get("maxspeed"))
				data["speed_kph"] = spd if spd is not None else default_speed
				fixed_spd += 1
			
		self.log(f"[Graph] Fixed {fixed_len} edges with missing length, {fixed_spd} edges with missing speed.")
		
		
	def add_charging_stations(self):
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
		chargers = ox.features_from_place(self.place, tags={"amenity": "charging_station"})
		
		stations = []
		for _, row in chargers.iterrows():
			if "geometry" not in row:
				continue
			geom = row.geometry
			if geom.is_empty:
				continue
			
			# Some are Points, others could be small polygons → use centroid
			pt = geom.centroid
			nearest_node = ox.distance.nearest_nodes(self.graph, pt.x, pt.y)
			
			# Tag this node as a charging station
			self.graph.nodes[nearest_node]["charging_station"] = True
			stations.append(nearest_node)
			
		print(f"[Graph] Added {len(stations)} charging stations.")
		return stations
		
	def add_buildings(self, showProgress = True, walking_speed_kph = 5):
		# ---- prepare buildings list ----
		self.log("[Graph] Fetching buildings...")
		buildings = ox.features_from_place(self.place, tags={"building": True})
		
		if buildings.empty:
			return "[Graph] No buildings found."
		if len(buildings) > self.maxBuildings:
			self.log(f"[Graph] Many buildings found, cutting to first {self.maxBuildings} buildings for performance")
			buildings = buildings.head(self.maxBuildings)
		crs = self.graph.graph.get("crs")
		if crs is None:
			# Assume CRS if no CRS is set
			self.graph.graph["crs"] = "EPSG:4326"
			crs = self.graph.graph["crs"]
			
		# Ensure CRS is a pyproj.CRS object
		if isinstance(crs, str):
			crs = pyproj.CRS.from_string(crs)
			
		if crs.is_geographic:
			self.graph = ox.project_graph(self.graph, to_crs="EPSG:3857")
			
		if buildings.crs.is_geographic:
			buildings = buildings.to_crs(self.graph.graph["crs"])
			
		# centroids as Points
		centroids = buildings.geometry.centroid
		xs = list(centroids.x)
		ys = list(centroids.y)
		
		# ---- use a frozen road-only graph for lookups (do not include building nodes) ----
		# copy only nodes/edges that look like road network (no building or connection types)
		# assume incoming G currently doesn't have building nodes; if it does, create a filtered copy.
		def is_road_edge(data):
			t = data.get("type")
			return t is None or str(t) not in ("building_link", "connection", "road_split")
		
		# create a lightweight graph for nearest-edge queries
		G_roads = nx.MultiDiGraph()
		for n, d in self.graph.nodes(data=True):
			if "building" in d or d.get("type") in ("building", "connection"):
				continue
			G_roads.add_node(n, **d)
		for u, v, k, data in self.graph.edges(keys=True, data=True):
			if is_road_edge(data):
				G_roads.add_edge(u, v, key=k, **data)
		# ensure CRS metadata is kept for osmnx functions
		if "crs" in self.graph.graph:
			G_roads.graph["crs"] = self.graph.graph["crs"]
			
		# ---- vectorized nearest-edge lookup (fast) ----
		# note: ox.distance.nearest_edges supports array inputs in modern osmnx
		try:
			nearest = ox.distance.nearest_edges(G_roads, xs, ys)
			# returns sequence of (u, v, key) tuples
		except Exception:
			# fallback: per-point nearest_edges (slower)
			nearest = [ox.distance.nearest_edges(G_roads, x, y) for x, y in zip(xs, ys)]
			
		self.log(f"List for storing nearest edges hat a length of {len(nearest)} with values like {nearest[len(nearest)-1]}")
		# Build list of candidate attachments (u,v,key, point index)
		edge_to_buildings = defaultdict(list)
		skipped_far = 0
		for i, (u, v, key) in enumerate(nearest):
			px, py = xs[i], ys[i]
			# fetch geometry of edge (u,v,key)
			ed = G_roads.get_edge_data(u, v, key)
			if ed is None:
				# weird fallback: skip
				skipped_far += 1
				continue
			line = ed.get("geometry")
			if line is None:
				# build line from node coords
				ux, uy = G_roads.nodes[u]["x"], G_roads.nodes[u]["y"]
				vx, vy = G_roads.nodes[v]["x"], G_roads.nodes[v]["y"]
				line = LineString([(ux, uy), (vx, vy)])
			# compute projected point and distance (along line)
			p = Point(px, py)
			proj_dist = line.project(p)         # distance along line (units same as G)
			proj_point = line.interpolate(proj_dist)
			euclidian = p.distance(proj_point)
			if euclidian <= self.maxSnapDistance:
				edge_id = (u, v, key)
				edge_to_buildings[edge_id].append((i, proj_dist, proj_point, euclidian))
			else:
				skipped_far += 1
				
		# ---- batch-split per edge ----
		next_node_id = max([n for n in self.graph.nodes]) + 1 if len(self.graph.nodes) > 0 and all(isinstance(n, int) for n in self.graph.nodes) else (len(self.graph.nodes) + 1)
		added_buildings = 0
		
		edge_items = list(edge_to_buildings.items())
		self.log(f"list edge_items hat a length of {len(edge_items)} with values like {edge_items[0]}")
		if showProgress:
			edge_items = tqdm(edge_items, desc="[Graph] Splitting edges and adding connections")
		split_success = 0
		split_exception = 0
		skipped_geom = 0
		bigger_counter = 0
		GeometryCollectionCounter = 0
		for (u, v, key), blist in edge_items:
			# sort buildings along edge by projection distance
			blist_sorted = sorted(blist, key=lambda x: x[1])
			# prepare list of shapely Points to split at
			split_points = [item[2] for item in blist_sorted]
			
			# get original edge data and geometry (prefer G edges if present)
			if self.graph.has_edge(u, v, key):
				orig_data = dict(self.graph.get_edge_data(u, v, key))  # copy
			else:
				orig_data = dict(G_roads.get_edge_data(u, v, key) or {})
				
			geom = orig_data.get("geometry")
			if geom is None:
				ux, uy = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
				vx, vy = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
				geom = LineString([(ux, uy), (vx, vy)])
				
			# split geometry at multi-points
			try:
				mp = MultiPoint(split_points)
				splitted = split(geom, mp).geoms
				split_success +=1
			except Exception:
				# fallback: skip splitting, just connect each building to nearest vertex
				splitted = [geom]
				split_exception +=1
				
			if isinstance(splitted, GeometrySequence):
				split_exception -=1
			elif isinstance(splitted, (MultiLineString, GeometryCollection)):
				# nothing meaningful to split, keep original edge
				print(splitted.geoms.geom_type)
				splitted = [geom for geom in splitted if isinstance(geom, LineString)]
				GeometryCollectionCounter +=1
			else:
				splitted = [splitted] if isinstance(splitted, LineString) else []
			# Only proceed if we really got 2 parts
			if len(splitted) == 2:
				part1, part2 = splitted
				# continue with edge replacement logic...
			elif len(splitted) >2:
				bigger_counter +=1
				#continue
			else:
				skipped_geom += 1
				continue
			# If splitted yields fewer pieces than number of split points + 1,
			# we'll fallback to sequential splitting (rare). For now assume OK.
			# Remove the original edge from the *main graph* (we will replace)
			if self.graph.has_edge(u, v, key):
				try:
					self.graph.remove_edge(u, v, key)
				except Exception:
					# multi-edges or already removed, ignore
					pass
					
			# Add splitted segments as new edges in-order between sequence of nodes:
			# We will create intermediate "connection nodes" at the exact split coordinates.
			# Build sequence of nodes: u -> conn1 -> conn2 -> ... -> v
			conn_nodes = []
			for pt in split_points:
				conn_nodes.append((next_node_id, pt))
				next_node_id += 1
				
			# helper to get coordinates
			def pt_coords(pt):
				return float(pt.x), float(pt.y)
			
			# add connection nodes to graph (they may already exist if multiple buildings project to same point)
			for nid, pt in conn_nodes:
				self.graph.add_node(nid, x=float(pt.x), y=float(pt.y), type="connection")
				
			# Now construct edges:
			# segments: [u -> conn1], [conn1 -> conn2], ..., [last_conn -> v]
			segment_nodes = [u] + [nid for nid, _ in conn_nodes] + [v]
			# segment geometries come from splitted list; if counts mismatch, create simple lines
			if len(splitted) == (len(segment_nodes) - 1):
				seg_geoms = splitted
			else:
				# fallback construct straight segments from coords
				seg_geoms = []
				for a, b in zip(segment_nodes[:-1], segment_nodes[1:]):
					seg_geoms.append(LineString([(self.graph.nodes[a]["x"], self.graph.nodes[a]["y"]), (self.graph.nodes[b]["x"], self.graph.nodes[b]["y"])]))
					
			# Add edges with original attributes preserved (but update geometry & length)
			for a, b, geom_seg in zip(segment_nodes[:-1], segment_nodes[1:], seg_geoms):
				seg_len = float(geom_seg.length)
				edge_attrs = orig_data.copy()
				edge_attrs["geometry"] = geom_seg
				edge_attrs["length"] = seg_len
				# ensure numeric speed
				if "speed_kph" not in edge_attrs or edge_attrs.get("speed_kph") is None:
					edge_attrs["speed_kph"] = float(edge_attrs.get("maxspeed", 30)) if isinstance(edge_attrs.get("maxspeed"), (int, float)) else 30.0
				self.graph.add_edge(a, b, **edge_attrs)
				# if original was two-way, add reverse with mirrored geometry
				if not edge_attrs.get("oneway", False):
					rev_attrs = edge_attrs.copy()
					rev_attrs["geometry"] = LineString(list(geom_seg.coords)[::-1])
					self.graph.add_edge(b, a, **rev_attrs)
					
			# Attach buildings to connection nodes:
			# we have blist_sorted -> each building maps to a split point -> we need the nearest connection node index
			for (b_index, proj_dist, proj_point, euclidian) in blist_sorted:
				# find closest connection node by distance to proj_point
				best_conn = None
				best_d = float("inf")
				for nid, pt in conn_nodes:
					d = proj_point.distance(pt)
					if d < best_d:
						best_conn = nid
						best_d = d
						
				if best_conn is None:
					# fallback attach to u
					attach_to = u
					attach_point = Point(self.graph.nodes[u]["x"], self.graph.nodes[u]["y"])
				else:
					attach_to = best_conn
					attach_point = Point(self.graph.nodes[best_conn]["x"], self.graph.nodes[best_conn]["y"])
					
				# add building node
				bnode = next_node_id
				next_node_id += 1
				bpt = Point(xs[b_index], ys[b_index])
				self.graph.add_node(bnode, x=float(bpt.x), y=float(bpt.y), building=True, type="building")
				
				# add bidirectional short walk edge
				dist = float(bpt.distance(attach_point))
				if dist <= self.maxSnapDistance and dist > 0.0:
					self.graph.add_edge(bnode, attach_to, length=dist, type="building_link", speed_kph=float(walking_speed_kph))
					self.graph.add_edge(attach_to, bnode, length=dist, type="building_link", speed_kph=float(walking_speed_kph))
					added_buildings += 1
				else:
					# skip if too far or zero
					pass
					
		# Final cleanup: ensure all edges have numeric length, speed
		for u, v, k, data in list(self.graph.edges(keys=True, data=True)):
			if data.get("length") is None:
				if "geometry" in data:
					data["length"] = float(data["geometry"].length)
				else:
					data["length"] = 1.0
			if data.get("speed_kph") is None:
				data["speed_kph"] = 30.0
				
		summary = {"added_buildings": added_buildings, "skipped_far": skipped_far, "edge_groups": len(edge_to_buildings), "skipped for skipped_geom": skipped_geom, "split_success": split_success, "split_exception": split_exception, "bigger_counter": bigger_counter, "GeometryCollectionCounter": GeometryCollectionCounter}
		return summary
	
	def parse_maxspeed(self, value):
		"""Try to extract a numeric speed (kph) from messy OSM maxspeed tags."""
		if value is None:
			return None
		if isinstance(value, (int, float)) and not math.isnan(value):
			return int(value)
		if isinstance(value, list):
			speeds = [self.parse_maxspeed(v) for v in value]
			speeds = [s for s in speeds if s is not None]
			return min(speeds) if speeds else None
		if isinstance(value, str):
			# Extract first number
			match = re.search(r"\d+", value)
			if match:
				return int(match.group())
		return None