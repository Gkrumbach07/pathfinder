#!/usr/bin/env python3

import rustworkx as rx
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import random
import contextily as ctx
from matplotlib.colors import to_rgba
from enum import Enum
import argparse
import math

# Constants
MAX_MATCH_DISTANCE = 0.25 * 1609.34  # 0.25 miles converted to meters
MAX_DISTANCE_PER_DAY = 12 * 1609.34  # 12 miles per day in meters
MIN_DISTANCE_PER_DAY = 5 * 1609.34  # 5 miles minimum per day
INITIAL_TEMPERATURE = 1000.0
COOLING_RATE = 0.995
MIN_TEMPERATURE = 0.01
TOTAL_NIGHTS = 3  # Number of nights
METERS_TO_MILES = 0.000621371  # Conversion factor from meters to miles

# Simulated Annealing parameters
MAX_ITERATIONS = 10000
POI_REWARD = 10.0
CAMP_PENALTY = 1000.0
DISTANCE_PENALTY = 100.0
OVERLAP_PENALTY = 200.0


def haversine(coord1, coord2):
    """
    Calculate the haversine distance between two (lon, lat) pairs.
    Returns distance in meters.
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


class PathType(Enum):
    LOOP = "loop"  # Start and end at same point
    POINT_TO_POINT = "point_to_point"  # Start and end at specific points
    OPEN_ENDED = "open_ended"  # Start at specific point, end anywhere in valid list


# Add to constants
VALID_ENDPOINTS = [2918151787, 2918151783, 2918151792]


class LoopPath:
    def __init__(self, points, camp_indices):
        """
        points: list of point indices forming the path
        camp_indices: list of indices into points where camping occurs
        """
        self.points = points
        self.camp_indices = camp_indices
        self._validate()

    def _validate(self):
        """Validate the path structure"""
        if not self.points:
            raise ValueError("Path cannot be empty")
        if self.points[0] != self.points[-1]:
            raise ValueError("Path must start and end at the same point")
        if not self.camp_indices:
            raise ValueError("Path must include at least one camping spot")
        if not all(0 <= idx < len(self.points) for idx in self.camp_indices):
            raise ValueError("Invalid camp indices")

    def copy(self):
        """Create a deep copy of the path"""
        return LoopPath(self.points.copy(), self.camp_indices.copy())

    def get_day_segments(self):
        """Split the path into day segments based on camping spots"""
        segments = []
        last_idx = 0

        # Handle each day's segment
        for camp_idx in self.camp_indices:
            segments.append(self.points[last_idx : camp_idx + 1])
            last_idx = camp_idx

        # Add final segment back to start
        segments.append(self.points[last_idx:])

        return segments

    def calculate_score(
        self, dist_matrix, poi_nodes, campsite_nodes, min_daily, max_daily
    ):
        """
        Calculate the score for this path based on multiple criteria:
        - Distance constraints
        - POI visits
        - Trail overlap
        - Camping spot distribution
        """
        score = 0.0
        segments = self.get_day_segments()

        # Calculate daily distances and penalties
        for segment in segments:
            if len(segment) < 2:
                continue

            # Calculate segment distance
            day_distance = sum(
                dist_matrix[segment[i]][segment[i + 1]] for i in range(len(segment) - 1)
            )

            # Distance constraint penalties
            if day_distance < min_daily:
                score -= (min_daily - day_distance) * DISTANCE_PENALTY
            elif day_distance > max_daily:
                score -= (day_distance - max_daily) * DISTANCE_PENALTY

        # Count POI visits
        poi_visits = sum(1 for point in self.points if point in poi_nodes)
        score += poi_visits * POI_REWARD

        # Check camping spot distribution
        camps_visited = sum(1 for point in self.points if point in campsite_nodes)
        if camps_visited < TOTAL_NIGHTS:
            score -= (TOTAL_NIGHTS - camps_visited) * CAMP_PENALTY

        # Calculate trail overlap penalty
        segment_usage = {}
        for i in range(len(self.points) - 1):
            segment = tuple(sorted([self.points[i], self.points[i + 1]]))
            segment_usage[segment] = segment_usage.get(segment, 0) + 1

        # Quadratic penalty for overlaps
        overlap_penalty = sum(
            (count - 1) ** 2 * OVERLAP_PENALTY for count in segment_usage.values()
        )
        score -= overlap_penalty

        return score

    def is_valid(self, dist_matrix, min_daily, max_daily):
        """Check if the path satisfies all constraints"""
        try:
            segments = self.get_day_segments()

            # Check daily distance constraints
            for segment in segments:
                if len(segment) < 2:
                    continue

                day_distance = sum(
                    dist_matrix[segment[i]][segment[i + 1]]
                    for i in range(len(segment) - 1)
                )

                if not (min_daily <= day_distance <= max_daily):
                    return False

            return True
        except Exception:
            return False


def create_path_graph(trails_gdf):
    """
    Create a graph from paths in a GeoDataFrame using projected CRS for accurate distances
    """
    # Create empty undirected graph
    G = rx.PyGraph()
    node_indices = {}  # Dictionary to store node indices by coordinates

    print("Converting paths to graph...")
    print(f"Original CRS: {trails_gdf.crs}")

    # Project to a suitable CRS for the area (UTM zone for the area)
    # For Isle Royale area (around 48°N, -89°W), use UTM zone 16N
    projected_gdf = trails_gdf.to_crs("EPSG:32616")  # UTM zone 16N
    print(f"Projected CRS: {projected_gdf.crs}")

    # Convert paths to graph
    for idx, row in projected_gdf.iterrows():
        # Extract coordinates from the linestring
        coords = list(row.geometry.coords)

        # Create nodes and edges for each segment
        for i in range(len(coords) - 1):
            # Store original unprojected coordinates for node data
            start_coord_orig = tuple(trails_gdf.iloc[idx].geometry.coords[i])
            end_coord_orig = tuple(trails_gdf.iloc[idx].geometry.coords[i + 1])

            # Add nodes if they don't exist (using original coords as keys)
            if start_coord_orig not in node_indices:
                node_indices[start_coord_orig] = G.add_node(start_coord_orig)
            if end_coord_orig not in node_indices:
                node_indices[end_coord_orig] = G.add_node(end_coord_orig)

            # Calculate distance using projected coordinates
            start_point = Point(coords[i])
            end_point = Point(coords[i + 1])
            distance = start_point.distance(end_point)  # Distance in meters

            # Add edge (undirected graph will handle both directions)
            G.add_edge(
                node_indices[start_coord_orig], node_indices[end_coord_orig], distance
            )

    print(f"Graph created with {G.num_nodes()} nodes and {G.num_edges()} edges")

    return G, node_indices


def find_nearest_node(point_coords, node_indices):
    """
    Find the nearest node to a point within the maximum distance threshold
    Returns (node_index, distance) or (None, None) if no node is within threshold
    """
    min_dist = float("inf")
    nearest_node = None

    for node_coord, node_idx in node_indices.items():
        # Calculate distance in meters
        dist = (
            np.sqrt(
                (point_coords[0] - node_coord[0]) ** 2
                + (point_coords[1] - node_coord[1]) ** 2
            )
            * 111139
        )  # Convert degrees to meters (approximate at mid-latitudes)

        # Only consider nodes within threshold
        if dist <= MAX_MATCH_DISTANCE:
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_idx

    if nearest_node is not None:
        return nearest_node, min_dist
    return None, None


def match_points_to_nodes(points_gdf, node_indices):
    """
    Match points to their nearest nodes within threshold
    Returns matched and unmatched points, preserving all original data
    """
    matched_points = []
    unmatched_points = []

    for idx, row in points_gdf.iterrows():
        point_coords = (row.geometry.x, row.geometry.y)
        nearest_node, distance = find_nearest_node(point_coords, node_indices)

        # Create a dictionary with all attributes from the GeoDataFrame
        point_data = {
            "point": point_coords,
            "node_idx": nearest_node if nearest_node is not None else None,
            "distance": distance if distance is not None else None,
        }

        # Add all other columns from the GeoDataFrame
        for column in points_gdf.columns:
            if column != "geometry":
                point_data[column] = row[column]

        if nearest_node is not None:
            matched_points.append(point_data)
        else:
            unmatched_points.append(point_data)

    return matched_points, unmatched_points


def plot_graph(
    G, node_indices, matched_pos, unmatched_pos, matched_camps, unmatched_camps
):
    """
    Plot the graph using matplotlib directly with POIs and campsites
    """
    plt.figure(figsize=(15, 10))

    # Plot graph edges
    for edge in G.edge_list():
        start_node = G.get_node_data(edge[0])
        end_node = G.get_node_data(edge[1])
        plt.plot(
            [start_node[0], end_node[0]],
            [start_node[1], end_node[1]],
            "gray",
            linewidth=0.5,
            alpha=0.5,
        )

    # Plot graph nodes
    coords = list(node_indices.keys())
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    plt.scatter(x_coords, y_coords, c="blue", s=10, alpha=0.6, label="Path Nodes")

    # Plot matched POIs
    if matched_pos:
        x_pos = [p["point"][0] for p in matched_pos]
        y_pos = [p["point"][1] for p in matched_pos]
        plt.scatter(x_pos, y_pos, c="green", s=100, alpha=0.6, label="Matched POIs")

    # Plot unmatched POIs
    if unmatched_pos:
        x_pos = [p["point"][0] for p in unmatched_pos]
        y_pos = [p["point"][1] for p in unmatched_pos]
        plt.scatter(x_pos, y_pos, c="grey", s=100, alpha=0.6, label="Unmatched POIs")

    # Plot matched campsites
    if matched_camps:
        x_camps = [p["point"][0] for p in matched_camps]
        y_camps = [p["point"][1] for p in matched_camps]
        plt.scatter(
            x_camps, y_camps, c="red", s=100, alpha=0.6, label="Matched Campsites"
        )

    # Plot unmatched campsites
    if unmatched_camps:
        x_camps = [p["point"][0] for p in unmatched_camps]
        y_camps = [p["point"][1] for p in unmatched_camps]
        plt.scatter(
            x_camps, y_camps, c="grey", s=100, alpha=0.6, label="Unmatched Campsites"
        )

    plt.title("Path Network with POIs and Campsites")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def create_distance_matrix(G, all_points, start_idx):
    """
    Create distance matrix and store paths for undirected graph
    Returns distances in miles
    """
    METERS_TO_MILES = 0.000621371  # More precise conversion factor

    print(f"\nCreating distance matrix for {len(all_points)} points...")

    # Get node indices from matched points
    subset_nodes = [p["node_idx"] for p in all_points]
    start_node = all_points[start_idx]["node_idx"]

    # Find all nodes connected to the start node
    print("Finding connected components from start node...")
    connected_nodes = set(rx.node_connected_component(G, start_node))

    # Filter points to only include connected nodes
    connected_points = []
    connected_indices = []
    for i, point in enumerate(all_points):
        if point["node_idx"] in connected_nodes:
            connected_points.append(point)
            connected_indices.append(i)

    print(f"Filtered {len(all_points) - len(connected_points)} disconnected points")
    print(f"Remaining connected points: {len(connected_points)}")

    # Update subset nodes to only include connected nodes
    subset_nodes = [p["node_idx"] for p in connected_points]

    # Initialize dictionaries
    subset_distances = {}
    subset_paths = {}

    # Compute shortest paths
    print("Computing shortest paths between points...")
    for i, node in enumerate(subset_nodes):
        if i % 10 == 0:
            print(f"Processing point {i}/{len(subset_nodes)}")

        try:
            # Get distances using float weights
            distances = rx.dijkstra_shortest_path_lengths(
                G, node, edge_cost_fn=lambda x: float(x)
            )
            # Convert distances to miles here
            subset_distances[node] = {
                k: v * METERS_TO_MILES for k, v in distances.items()
            }

            # Print some sample distances for verification
            if i == 0:  # Only for first node
                print("\nSample distances from first node:")
                for dest, dist in list(distances.items())[:5]:
                    print(
                        f"Distance: {dist:.2f} meters = {dist * METERS_TO_MILES:.2f} miles"
                    )

            # Get complete paths between points
            paths = rx.dijkstra_shortest_paths(G, node, weight_fn=lambda x: float(x))
            subset_paths[node] = paths

        except Exception as e:
            print(f"Warning: Error computing paths from node {node}: {str(e)}")
            continue

    # Create distance matrix (will now contain miles)
    n = len(connected_points)
    dist_matrix = np.full((n, n), np.inf)

    # Fill distance matrix
    for i, u in enumerate(subset_nodes):
        for j, v in enumerate(subset_nodes):
            if i == j:
                dist_matrix[i][j] = 0.0
            else:
                if u in subset_distances and v in subset_distances[u]:
                    dist_matrix[i][j] = float(
                        subset_distances[u][v]
                    )  # Already in miles

    # Create mappings
    point_to_matrix = {p["node_idx"]: i for i, p in enumerate(connected_points)}
    matrix_to_point = {i: p["node_idx"] for i, p in enumerate(connected_points)}

    # Map start_idx to new index in connected points
    new_start_idx = connected_indices.index(start_idx)

    # Verify matrix
    finite_distances = dist_matrix[~np.isinf(dist_matrix)]
    if len(finite_distances) > 0:
        print("\nVerification of distance matrix:")
        print(f"Number of finite distances: {len(finite_distances)}")
        print(
            f"Distance range: {np.min(finite_distances):.2f} to {np.max(finite_distances):.2f} miles"
        )
        print(f"Number of infinite distances: {np.sum(np.isinf(dist_matrix))}")
        print(f"Matrix shape: {dist_matrix.shape}")

    return (
        dist_matrix,
        subset_paths,
        point_to_matrix,
        matrix_to_point,
        connected_points,
        new_start_idx,
    )


def find_start_node(matched_points, target_osm_id="2918151792"):
    """Find the starting node index"""
    for i, point in enumerate(matched_points):
        if point.get("osm_id") == target_osm_id:
            print(f"Found starting point at index {i}")
            return i
    # If not found, return the first point
    return 0


def generate_initial_loop(
    dist_matrix,
    start_idx,
    all_points,
    matched_camps,
    total_nights,
    max_day_distance=MAX_DISTANCE_PER_DAY,
    min_day_distance=MIN_DISTANCE_PER_DAY,
):
    """Generate initial loop by ensuring camping spots and distance constraints"""
    print("\nGenerating initial path...")

    # Get indices of all campsites
    campsite_indices = set(range(len(all_points) - len(matched_camps), len(all_points)))

    # Start with the start point
    path = [start_idx]
    camp_indices = []

    # Track available points
    available_points = set(range(len(all_points)))
    available_points.remove(start_idx)

    # Calculate target total distance (aim for middle of min/max range)
    target_total_distance = (min_day_distance + max_day_distance) / 2 * total_nights
    current_distance = 0
    current = start_idx

    print(f"Target total distance: {target_total_distance * METERS_TO_MILES:.2f} miles")

    # First, try to add camping spots
    camps_added = 0
    while camps_added < total_nights and available_points:
        # Find nearest campsite
        nearest_camp = None
        min_dist = float("inf")

        for point in available_points:
            if point in campsite_indices:
                dist = dist_matrix[current][point]
                if dist < min_dist and dist <= max_day_distance:
                    min_dist = dist
                    nearest_camp = point

        if nearest_camp is None:
            print("Could not find suitable campsite")
            break

        # Add campsite to path
        path.append(nearest_camp)
        camp_indices.append(len(path) - 1)
        available_points.remove(nearest_camp)
        current = nearest_camp
        current_distance += min_dist
        camps_added += 1

        print(f"Added campsite {camps_added}/{total_nights}")

    # Now fill in the gaps between campsites with POIs
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        current_dist = dist_matrix[start][end]

        # Try to add POIs between campsites
        while current_dist < max_day_distance and available_points:
            # Find nearest POI that won't exceed max distance
            nearest_poi = None
            min_dist = float("inf")

            for point in available_points:
                if point not in campsite_indices:
                    dist = dist_matrix[current][point]
                    if dist < min_dist and current_dist + dist <= max_day_distance:
                        min_dist = dist
                        nearest_poi = point

            if nearest_poi is None:
                break

            # Add POI to path
            path.insert(i + 1, nearest_poi)
            available_points.remove(nearest_poi)
            current_dist += min_dist
            current = nearest_poi

            # Update camp indices
            for j in range(len(camp_indices)):
                if camp_indices[j] > i:
                    camp_indices[j] += 1

    # Finally, ensure we return to start
    if path[-1] != start_idx:
        path.append(start_idx)

    # Create and validate the path
    loop_path = LoopPath(path, camp_indices)

    # Print path info
    print(f"\nInitial path statistics:")
    print(f"- Points: {len(path)}")
    print(f"- Total distance: {current_distance * METERS_TO_MILES:.2f} miles")
    print(f"- Camping spots: {len(camp_indices)}")

    return loop_path


def fill_intermediate_points(path, paths_dict, matrix_to_point, all_points):
    """
    Fill in intermediate points along the path using stored paths
    """
    enhanced_path = []

    for i in range(len(path) - 1):
        current = path[i]
        next_point = path[i + 1]

        # Add current point
        enhanced_path.append(current)

        try:
            # Get the complete path between points
            if current in paths_dict and next_point in paths_dict[current]:
                intermediate_path = paths_dict[current][next_point]
                # Add all intermediate points (excluding start and end)
                enhanced_path.extend(intermediate_path[1:-1])

        except Exception as e:
            print(
                f"Warning: Could not fill points between {current} and {next_point}: {str(e)}"
            )
            continue

    # Add final point
    enhanced_path.append(path[-1])

    return enhanced_path


def calculate_path_score(
    loop_path,
    dist_matrix,
    matched_points,
    G,
    total_nights,
    matched_camps,
    max_day_distance=MAX_DISTANCE_PER_DAY,
):
    """Calculate score with better balanced penalties"""
    points = loop_path.points
    camp_indices = loop_path.camp_indices

    # Penalize missing camping spots instead of rejecting
    camping_penalty = abs(len(camp_indices) - total_nights)
    # Basic metrics
    unique_points = len(set(points))

    # Calculate daily distances and penalties
    day_segments = loop_path.get_day_segments()
    daily_distances = []
    distance_penalties = 0

    for segment in day_segments:
        if len(segment) < 2:
            continue

        # Calculate segment distance using distance matrix
        day_distance = sum(
            dist_matrix[segment[i]][segment[i + 1]] for i in range(len(segment) - 1)
        )
        daily_distances.append(day_distance)

        # Harsh penalty for exceeding max distance
        if day_distance > max_day_distance:
            excess = day_distance - max_day_distance
            distance_penalties += (excess / max_day_distance) ** 2 * 1000
        elif day_distance < max_day_distance * 0.2:  # Penalize very short days
            shortfall = max_day_distance * 0.2 - day_distance
            distance_penalties += shortfall * 10

    # Calculate path overlap penalty using distance matrix
    # Count how many times each segment is used
    segment_usage = {}
    for i in range(len(points) - 1):
        segment = tuple(sorted([points[i], points[i + 1]]))
        segment_usage[segment] = segment_usage.get(segment, 0) + 1

    # Quadratic penalty for overlaps to more strongly discourage multiple uses
    overlap_penalty = sum((count - 1) ** 2 * 100 for count in segment_usage.values())

    # Additional penalty for immediate out-and-back segments
    backtrack_penalty = 0
    for i in range(len(points) - 2):
        if points[i] == points[i + 2]:  # Immediate backtrack
            backtrack_penalty += 200

    # Check for balanced daily distances
    if len(daily_distances) > 1:
        avg_distance = sum(daily_distances) / len(daily_distances)
        balance_penalty = sum(abs(d - avg_distance) for d in daily_distances)
    else:
        balance_penalty = 0

    # Scoring weights
    point_weight = 2.0
    distance_penalty_weight = 1.0
    overlap_weight = 20.0
    balance_weight = 2.0
    backtrack_weight = 10.0  # New weight for backtrack penalty

    score = (
        point_weight * unique_points
        - distance_penalty_weight * distance_penalties
        - overlap_weight * overlap_penalty
        - balance_weight * balance_penalty
        - backtrack_weight * backtrack_penalty
        - camping_penalty
    )

    return score


def optimize_loop(
    initial_path,
    dist_matrix,
    matched_points,
    G,
    total_nights,
    matched_camps,
    paths_dict,
    matrix_to_point,
    max_iterations=MAX_ITERATIONS,
    max_day_distance=MAX_DISTANCE_PER_DAY,
    min_day_distance=MIN_DISTANCE_PER_DAY,
):
    """Optimize loop using simulated annealing with improved move operators"""
    print("\nOptimize Loop Starting State:")
    print(f"Total points: {len(matched_points)}")
    print(f"Total campsites: {len(matched_camps)}")
    print(f"Required nights: {total_nights}")

    # Get indices of all campsites and POIs
    campsite_indices = set(
        range(len(matched_points) - len(matched_camps), len(matched_points))
    )
    poi_indices = set(range(len(matched_points) - len(matched_camps)))

    current_path = initial_path.copy()
    current_score = current_path.calculate_score(
        dist_matrix, poi_indices, campsite_indices, min_day_distance, max_day_distance
    )
    best_score = current_score
    best_path = current_path.copy()

    temperature = INITIAL_TEMPERATURE
    iteration = 0
    last_improvement = 0

    # Keep track of available points for insertion
    all_points = set(range(len(matched_points)))
    points_in_path = set(current_path.points)
    available_points = list(all_points - points_in_path)

    print(f"Starting optimization with {max_iterations} iterations")
    print(f"Initial path score: {current_score:.2f}")

    while temperature > MIN_TEMPERATURE and iteration < max_iterations:
        iteration += 1

        # Choose move type based on current state
        possible_moves = []

        # Add regular moves
        if len(current_path.points) > 3:
            possible_moves.append("two_opt")
            possible_moves.append("swap")
        if available_points:
            possible_moves.append("insert")
        if len(current_path.points) > 4:
            possible_moves.append("delete")

        # Add camping-related moves
        if len(current_path.camp_indices) < total_nights:
            possible_moves.append("add_camp")
        elif len(current_path.camp_indices) == total_nights:
            possible_moves.append("shuffle_camps")

        if not possible_moves:
            print("No valid moves available")
            break

        move_type = random.choice(possible_moves)
        neighbor = current_path.copy()

        # Apply selected move
        if move_type == "two_opt":
            # Two-opt move: reverse a segment of the path
            if len(neighbor.points) > 3:
                i = random.randint(1, len(neighbor.points) - 3)
                j = random.randint(i + 1, len(neighbor.points) - 2)
                neighbor.points[i : j + 1] = neighbor.points[i : j + 1][::-1]
                # Update camp indices
                neighbor.camp_indices = [
                    j - (idx - i) if i <= idx <= j else idx
                    for idx in neighbor.camp_indices
                ]

        elif move_type == "swap":
            # Swap two non-adjacent points
            if len(neighbor.points) > 3:
                i = random.randint(1, len(neighbor.points) - 2)
                j = random.randint(1, len(neighbor.points) - 2)
                neighbor.points[i], neighbor.points[j] = (
                    neighbor.points[j],
                    neighbor.points[i],
                )
                # Update camp indices
                neighbor.camp_indices = [
                    j if idx == i else i if idx == j else idx
                    for idx in neighbor.camp_indices
                ]

        elif move_type == "insert":
            # Insert a new point
            if available_points:
                point_to_insert = random.choice(available_points)
                insert_pos = random.randint(1, len(neighbor.points) - 1)
                neighbor.points.insert(insert_pos, point_to_insert)
                # Update camp indices
                neighbor.camp_indices = [
                    idx + 1 if idx >= insert_pos else idx
                    for idx in neighbor.camp_indices
                ]

        elif move_type == "delete":
            # Delete a point that isn't a camping spot
            deletable_positions = [
                i
                for i in range(1, len(neighbor.points) - 1)
                if i not in neighbor.camp_indices
            ]
            if deletable_positions:
                delete_pos = random.choice(deletable_positions)
                neighbor.points.pop(delete_pos)
                # Update camp indices
                neighbor.camp_indices = [
                    idx - 1 if idx > delete_pos else idx
                    for idx in neighbor.camp_indices
                ]

        elif move_type == "add_camp":
            # Add a camping spot
            available_campsites = [p for p in available_points if p in campsite_indices]
            if available_campsites:
                campsite_to_insert = random.choice(available_campsites)
                insert_pos = random.randint(1, len(neighbor.points) - 1)
                neighbor.points.insert(insert_pos, campsite_to_insert)
                neighbor.camp_indices.append(insert_pos)
                neighbor.camp_indices.sort()

        elif move_type == "shuffle_camps":
            # Move a camping spot to a new position
            if len(neighbor.camp_indices) > 1:
                camp_idx = random.randint(0, len(neighbor.camp_indices) - 1)
                old_pos = neighbor.camp_indices[camp_idx]
                new_pos = random.randint(1, len(neighbor.points) - 1)
                if new_pos not in neighbor.camp_indices:
                    neighbor.camp_indices[camp_idx] = new_pos
                    neighbor.camp_indices.sort()

        # Fill in intermediate points
        enhanced_points = fill_intermediate_points(
            neighbor.points, paths_dict, matrix_to_point, matched_points
        )
        neighbor = LoopPath(enhanced_points, neighbor.camp_indices)

        # Calculate new score
        neighbor_score = neighbor.calculate_score(
            dist_matrix,
            poi_indices,
            campsite_indices,
            min_day_distance,
            max_day_distance,
        )

        # Accept or reject new solution
        delta = neighbor_score - current_score
        accept = False

        if delta > 0:
            accept = True
        elif delta == 0:
            # Accept lateral moves with high probability early in search
            accept = random.random() < temperature / INITIAL_TEMPERATURE
        else:
            # Accept worse moves with decreasing probability
            accept = random.random() < np.exp(delta / temperature)

        if accept:
            current_path = neighbor
            current_score = neighbor_score

            if current_score > best_score:
                best_score = current_score
                best_path = current_path.copy()
                last_improvement = iteration
                print(f"\nNew best solution found at iteration {iteration}:")
                print(f"Score: {best_score:.2f}")
                print(f"Move type: {move_type}")

        temperature *= COOLING_RATE

        # Early stopping if no improvement for a while
        if iteration - last_improvement > max_iterations // 4:
            print("\nStopping early due to lack of improvement")
            break

    print("=== Path Generation Complete ===")
    print("Final path statistics:")
    print(f"- Points in path: {len(best_path.points)}")
    print(f"- Unique points: {len(set(best_path.points))}")
    print(f"- Camping spots: {len(best_path.camp_indices)}")
    print(f"- Final score: {best_score:.2f}")

    return best_path, best_score


def generate_path(
    G,
    matched_points,
    matched_camps,
    path_type=PathType.LOOP,
    max_day_distance=MAX_DISTANCE_PER_DAY,
    total_nights=TOTAL_NIGHTS,
    start_point=None,
    end_point=None,
    max_iterations=10000,
):
    """Main function to generate and optimize a path"""
    print(f"\n=== Starting {path_type.value.title()} Generation ===")

    # Combine POIs and campsites
    all_points = matched_points + matched_camps
    print(f"Total points to consider: {len(all_points)}")
    print(f"- POIs: {len(matched_points)}")
    print(f"- Campsites: {len(matched_camps)}")

    # Convert start_point to the format expected by find_nearest_node
    node_indices = {tuple(p["point"]): i for i, p in enumerate(all_points)}
    start_node, start_dist = find_nearest_node(
        (start_point[1], start_point[0]),  # Convert [lat,lon] to [lon,lat]
        node_indices,
    )
    if start_node is None:
        raise ValueError("Could not find valid start point")
    start_idx = start_node
    print(f"Found start point at distance: {start_dist * METERS_TO_MILES:.2f} miles")

    # Create distance matrix using start point
    print("\nGenerating distance and successor matrices...")
    (
        dist_matrix,
        paths_dict,
        point_to_matrix,
        matrix_to_point,
        connected_points,
        new_start_idx,
    ) = create_distance_matrix(G, all_points, start_idx)

    # Update indices for connected points
    start_idx = new_start_idx

    # Find end point if needed (using connected points)
    end_idx = None
    if path_type == PathType.POINT_TO_POINT:
        end_idx = find_nearest_node_to_coords(end_point, connected_points)
    elif path_type == PathType.LOOP:
        end_idx = start_idx

    if path_type == PathType.POINT_TO_POINT and end_idx is None:
        raise ValueError("Could not find valid end point")

    # Rest of the function remains the same but uses connected_points instead of all_points
    start_point_data = connected_points[start_idx]
    print(
        f"Starting point: ({start_point_data['point'][0]:.6f}, {start_point_data['point'][1]:.6f})"
    )

    # Generate initial path with connected points
    print("\nGenerating initial path...")
    initial_loop = generate_initial_loop(
        dist_matrix,
        start_idx,
        connected_points,
        matched_camps,
        total_nights,
        max_day_distance,
        min_day_distance=MIN_DISTANCE_PER_DAY,
    )

    # Check if we got a valid initial path
    if initial_loop is None:
        print("Could not generate a valid initial path")
        return None, None, None

    initial_distance = sum(
        dist_matrix[initial_loop.points[i]][initial_loop.points[i + 1]]
        for i in range(len(initial_loop.points) - 1)
    )

    print(f"Initial path statistics:")
    print(f"- Points in path: {len(initial_loop.points)}")
    print(f"- Unique points: {len(set(initial_loop.points))}")
    print(f"- Total distance: {initial_distance:.2f} miles")
    print(f"- Camping spots: {len(initial_loop.camp_indices)}")

    # Fill in intermediate points using paths
    print("\nFilling in intermediate points...")
    enhanced_path = fill_intermediate_points(
        initial_loop.points, paths_dict, matrix_to_point, connected_points
    )
    enhanced_loop = LoopPath(enhanced_path, initial_loop.camp_indices)

    enhanced_distance = sum(
        dist_matrix[enhanced_loop.points[i]][enhanced_loop.points[i + 1]]
        for i in range(len(enhanced_loop.points) - 1)
    )
    print(f"Enhanced path statistics:")
    print(f"- Points in path: {len(enhanced_loop.points)}")
    print(f"- Unique points: {len(set(enhanced_loop.points))}")
    print(f"- Total distance: {enhanced_distance:.2f} miles")
    print(f"- Camping spots: {len(enhanced_loop.camp_indices)}")

    # Optimize path
    print("\nStarting path optimization...")
    try:
        best_path_and_score = optimize_loop(
            enhanced_loop,
            dist_matrix,
            connected_points,
            G,
            total_nights,
            matched_camps,
            paths_dict,
            matrix_to_point,
            max_iterations=max_iterations,
            max_day_distance=max_day_distance,
            min_day_distance=MIN_DISTANCE_PER_DAY,
        )
        best_path, best_score = best_path_and_score
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        print(f"Current path state: {enhanced_loop.points}")
        raise

    return (
        best_path,
        connected_points,
        dist_matrix,
        paths_dict,
    )  # Return as tuple


def generate_route_description(
    path, all_points, dist_matrix, optimized_loop, path_type
):
    """
    Generate a human-readable description of the route organized by days
    """
    description = []
    total_distance = 0

    # Get day segments from the loop path
    day_segments = optimized_loop.get_day_segments()

    # Start description
    start_point = all_points[path[0]]
    start_name = start_point.get("name", "Unknown Location")
    description.append(f"Your journey begins at {start_name}.\n")

    # Generate description for each day
    for day_num, segment in enumerate(day_segments, 1):
        day_distance = 0
        description.append(f"Day {day_num}:")

        # Describe each segment of this day's journey
        for i in range(len(segment) - 1):
            current = all_points[segment[i]]
            next_point = all_points[segment[i + 1]]

            # Get names or default descriptions
            current_name = current.get("name", "Unknown Location")
            next_name = next_point.get("name", "Unknown Location")

            # Calculate segment distance
            segment_distance = dist_matrix[segment[i]][segment[i + 1]]
            day_distance += segment_distance
            total_distance += segment_distance

            description.append(
                f"  - Walk {segment_distance:.1f} miles from {current_name} to {next_name}"
            )

        # Add camping information if this isn't the last day
        if day_num <= len(optimized_loop.camp_indices):
            camp_idx = optimized_loop.camp_indices[day_num - 1]
            camp_point = all_points[optimized_loop.points[camp_idx]]
            camp_name = camp_point.get("name", "Unknown Location")
            description.append(f"  * Camp at {camp_name}")

        description.append(f"  Total day's distance: {day_distance:.1f} miles\n")

    # Add summary
    description.append("Summary:")
    description.append(f"- Total journey distance: {total_distance:.1f} miles")
    description.append(f"- Days of hiking: {len(day_segments)}")
    description.append(f"- Nights of camping: {len(optimized_loop.camp_indices)}")
    description.append(f"- Total points of interest visited: {len(set(path))-1}")

    # Add path type specific information
    if path_type == PathType.LOOP:
        description.append("- Path type: Loop (returns to starting point)")
    elif path_type == PathType.POINT_TO_POINT:
        end_point = all_points[path[-1]]
        end_name = end_point.get("name", "Unknown Location")
        description.append(f"- Path type: Point-to-point (ends at {end_name})")
    else:  # OPEN_ENDED
        end_point = all_points[path[-1]]
        end_name = end_point.get("name", "Unknown Location")
        description.append(f"- Path type: Open-ended (ended at {end_name})")

    return "\n".join(description)


def find_nearest_node_to_coords(point: tuple[float, float], all_points) -> int:
    """Find the index of the nearest node to the given coordinates using match_points_to_nodes"""
    # Create a single-row GeoDataFrame for the target point
    point_gdf = gpd.GeoDataFrame(
        geometry=[
            Point(point[1], point[0])
        ],  # Convert [lat,lon] to [lon,lat] for Point
        crs="EPSG:4326",
    )

    # Use existing match_points_to_nodes function
    matched_points, unmatched_points = match_points_to_nodes(
        point_gdf,
        {
            tuple(p["point"][::-1]): i  # Reverse point order to match expected format
            for i, p in enumerate(all_points)
        },
    )

    if not matched_points:
        raise ValueError(f"Could not find any nodes near coordinates {point}")

    nearest_idx = matched_points[0]["node_idx"]
    distance = matched_points[0]["distance"]

    print(f"Target point [lat, lon]: ({point[0]:.6f}, {point[1]:.6f})")
    print(
        f"Nearest node [lat, lon]: ({all_points[nearest_idx]['point'][0]:.6f}, {all_points[nearest_idx]['point'][1]:.6f})"
    )
    print(f"Distance: {distance * METERS_TO_MILES:.2f} miles")

    return nearest_idx
