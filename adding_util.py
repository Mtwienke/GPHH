#!/usr/bin/env python3

import math
import itertools
from collections import defaultdict

import osmnx as ox
import networkx as nx
import shapely
from shapely.geometry import Point, LineString, MultiPoint, GeometryCollection, MultiLineString 
from shapely.geometry.base import GeometrySequence
from shapely.ops import split
from tqdm import tqdm
import pyproj
import utils

def add_buildings_batch(
	G,
	place,
	max_buildings: int | None = 1000,
	max_snap_dist: float = 1000.0,
	show_progress: bool = True,
	walking_speed_kph: float = 5.0,
	logger = utils.Logger()
):
	"""
	Batch-attach buildings to road edges: vectorized nearest-edge lookup,
	group by edge, split each edge once, add connection nodes and building links.

	Parameters
	----------
	G : networkx.MultiDiGraph
		Projected road graph (units in meters). Must contain edge 'geometry'.
	place : string
		Name of the place of the graph
	max_buildings : int or None
		Limit number of buildings to process (None = all).
	max_snap_dist : float
		Maximum snapping distance in meters. Buildings further away will be skipped.
	show_progress : bool
		Whether to show tqdm progress bar.
	walking_speed_kph : float
		Speed assigned to building links (walking).

	Returns
	-------
	G_new : networkx.MultiDiGraph
		Graph with building nodes, connection nodes, and updated edges.
	summary : dict
		Stats about processed/skipped counts.
	"""
	# ---- prepare buildings list ----
	print("[Graph] Fetching buildings...")
	buildings = ox.features_from_place(place, tags={"building": True})
	
	if buildings.empty:
		logger.log("[Graph] No buildings found.")
		return G
	if len(buildings) > max_buildings:
		logger.log(f"[Graph] Many buildings found, cutting to first {max_buildings} buildings for performance")
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
	for n, d in G.nodes(data=True):
		if "building" in d or d.get("type") in ("building", "connection"):
			continue
		G_roads.add_node(n, **d)
	for u, v, k, data in G.edges(keys=True, data=True):
		if is_road_edge(data):
			G_roads.add_edge(u, v, key=k, **data)
	# ensure CRS metadata is kept for osmnx functions
	if "crs" in G.graph:
		G_roads.graph["crs"] = G.graph["crs"]
			
	# ---- vectorized nearest-edge lookup (fast) ----
	# note: ox.distance.nearest_edges supports array inputs in modern osmnx
	try:
		nearest = ox.distance.nearest_edges(G_roads, xs, ys)
		# returns sequence of (u, v, key) tuples
	except Exception:
		# fallback: per-point nearest_edges (slower)
		nearest = [ox.distance.nearest_edges(G_roads, x, y) for x, y in zip(xs, ys)]
	
	logger.log(f"List for storing nearest edges hat a length of {len(nearest)} with values like {nearest[len(nearest)-1]}")
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
		if euclidian <= max_snap_dist:
			edge_id = (u, v, key)
			edge_to_buildings[edge_id].append((i, proj_dist, proj_point, euclidian))
		else:
			skipped_far += 1
			
	# ---- batch-split per edge ----
	next_node_id = max([n for n in G.nodes]) + 1 if len(G.nodes) > 0 and all(isinstance(n, int) for n in G.nodes) else (len(G.nodes) + 1)
	added_buildings = 0
	
	edge_items = list(edge_to_buildings.items())
	logger.log(f"list edge_items hat a length of {len(edge_items)} with values like {edge_items[0]}")
	if show_progress:
		edge_items = tqdm(edge_items, desc="[Graph] Splitting edges and adding connections")
	split_success = 0
	split_exception = 0
	skipped_geom = 0
	bigger_counter = 0
	print(shapely.__version__)
	GeometryCollectionCounter = 0
	for (u, v, key), blist in edge_items:
		# sort buildings along edge by projection distance
		blist_sorted = sorted(blist, key=lambda x: x[1])
		# prepare list of shapely Points to split at
		split_points = [item[2] for item in blist_sorted]
		
		# get original edge data and geometry (prefer G edges if present)
		if G.has_edge(u, v, key):
			orig_data = dict(G.get_edge_data(u, v, key))  # copy
		else:
			orig_data = dict(G_roads.get_edge_data(u, v, key) or {})
			
		geom = orig_data.get("geometry")
		if geom is None:
			ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
			vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
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
		if G.has_edge(u, v, key):
			try:
				G.remove_edge(u, v, key)
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
			G.add_node(nid, x=float(pt.x), y=float(pt.y), type="connection")
			
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
				seg_geoms.append(LineString([(G.nodes[a]["x"], G.nodes[a]["y"]), (G.nodes[b]["x"], G.nodes[b]["y"])]))
				
		# Add edges with original attributes preserved (but update geometry & length)
		for a, b, geom_seg in zip(segment_nodes[:-1], segment_nodes[1:], seg_geoms):
			seg_len = float(geom_seg.length)
			edge_attrs = orig_data.copy()
			edge_attrs["geometry"] = geom_seg
			edge_attrs["length"] = seg_len
			# ensure numeric speed
			if "speed_kph" not in edge_attrs or edge_attrs.get("speed_kph") is None:
				edge_attrs["speed_kph"] = float(edge_attrs.get("maxspeed", 30)) if isinstance(edge_attrs.get("maxspeed"), (int, float)) else 30.0
			G.add_edge(a, b, **edge_attrs)
			# if original was two-way, add reverse with mirrored geometry
			if not edge_attrs.get("oneway", False):
				rev_attrs = edge_attrs.copy()
				rev_attrs["geometry"] = LineString(list(geom_seg.coords)[::-1])
				G.add_edge(b, a, **rev_attrs)
				
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
				attach_point = Point(G.nodes[u]["x"], G.nodes[u]["y"])
			else:
				attach_to = best_conn
				attach_point = Point(G.nodes[best_conn]["x"], G.nodes[best_conn]["y"])
				
			# add building node
			bnode = next_node_id
			next_node_id += 1
			bpt = Point(xs[b_index], ys[b_index])
			G.add_node(bnode, x=float(bpt.x), y=float(bpt.y), building=True, type="building")
			
			# add bidirectional short walk edge
			dist = float(bpt.distance(attach_point))
			if dist <= max_snap_dist and dist > 0.0:
				G.add_edge(bnode, attach_to, length=dist, type="building_link", speed_kph=float(walking_speed_kph))
				G.add_edge(attach_to, bnode, length=dist, type="building_link", speed_kph=float(walking_speed_kph))
				added_buildings += 1
			else:
				# skip if too far or zero
				pass
				
	# Final cleanup: ensure all edges have numeric length, speed
	for u, v, k, data in list(G.edges(keys=True, data=True)):
		if data.get("length") is None:
			if "geometry" in data:
				data["length"] = float(data["geometry"].length)
			else:
				data["length"] = 1.0
		if data.get("speed_kph") is None:
			data["speed_kph"] = 30.0
			
	summary = {"added_buildings": added_buildings, "skipped_far": skipped_far, "edge_groups": len(edge_to_buildings), "skipped for skipped_geom": skipped_geom, "split_success": split_success, "split_exception": split_exception, "bigger_counter": bigger_counter, "GeometryCollectionCounter": GeometryCollectionCounter}
	return G, summary
