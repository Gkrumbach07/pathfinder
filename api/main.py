from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import (
    PathRequest,
    PathResponse,
    PathType,
    GeoJSONLineString,
    DailySegment,
    SegmentDistance,
)
from .osm_fetcher import OSMFetcher
from .pathfinder import create_path_graph, generate_path, match_points_to_nodes
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OSM fetcher
osm_fetcher = OSMFetcher()


@app.post("/generate-path", response_model=PathResponse)
async def generate_hiking_path(request: PathRequest):
    try:
        # Calculate bounds
        buffer = request.max_day_distance * request.nights / 2 * 1.2
        bounds = calculate_bounds(request, buffer_miles=buffer)

        # Fetch OSM data
        trails_gdf, pois_gdf, camps_gdf = osm_fetcher.get_data_for_bounds(bounds)

        # Validate data
        if trails_gdf.empty:
            raise HTTPException(status_code=400, detail="No trails found in the area")
        if camps_gdf.empty:
            raise HTTPException(
                status_code=400, detail="No campsites found in the area"
            )

        print(f"Found {len(trails_gdf)} trail segments")
        print(f"Found {len(pois_gdf)} points of interest")
        print(f"Found {len(camps_gdf)} campsites")

        # Create graph and match points
        try:
            G, node_indices = create_path_graph(trails_gdf)
        except Exception as e:
            print(f"Error creating graph: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error creating graph: {str(e)}"
            )

        try:
            matched_pos, _ = match_points_to_nodes(pois_gdf, node_indices)
            matched_camps, _ = match_points_to_nodes(camps_gdf, node_indices)
        except Exception as e:
            print(f"Error matching points: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error matching points: {str(e)}"
            )

        if not matched_camps:
            raise HTTPException(
                status_code=400,
                detail="No campsites could be matched to the trail network",
            )

        # Generate path
        path_result = generate_path(
            G,
            matched_pos,
            matched_camps,
            path_type=request.path_type,
            max_day_distance=request.max_day_distance,
            total_nights=request.nights,
            start_point=request.start_point,
            end_point=request.end_point,
        )

        if path_result is None:
            raise HTTPException(status_code=400, detail="Could not generate valid path")

        optimized_path, all_points, dist_matrix, paths_dict = path_result

        # Convert trail network to GeoJSON
        trail_network = [
            {
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in row.geometry.coords],
            }
            for _, row in trails_gdf.iterrows()
        ]

        # Create response
        response = PathResponse(
            path=GeoJSONLineString(
                type="LineString",
                coordinates=[
                    [float(x), float(y)]
                    for i in range(len(optimized_path.points) - 1)
                    for node in paths_dict[
                        all_points[optimized_path.points[i]]["node_idx"]
                    ][all_points[optimized_path.points[i + 1]]["node_idx"]]
                    for x, y in [G.get_node_data(node)]
                ],
            ),
            daily_segments=[
                DailySegment(
                    path=GeoJSONLineString(
                        type="LineString",
                        coordinates=[
                            [float(x), float(y)]
                            for i in range(len(segment) - 1)
                            for node in paths_dict[all_points[segment[i]]["node_idx"]][
                                all_points[segment[i + 1]]["node_idx"]
                            ]
                            for x, y in [G.get_node_data(node)]
                        ],
                    ),
                    points=[all_points[point_idx] for point_idx in segment],
                    distances=[
                        SegmentDistance(
                            from_point=all_points[segment[i]],
                            to_point=all_points[segment[i + 1]],
                            distance=float(dist_matrix[segment[i]][segment[i + 1]]),
                        )
                        for i in range(len(segment) - 1)
                    ],
                    total_distance=float(
                        sum(
                            dist_matrix[segment[i]][segment[i + 1]]
                            for i in range(len(segment) - 1)
                        )
                    ),
                    camping_spot=(
                        all_points[
                            optimized_path.points[optimized_path.camp_indices[day_idx]]
                        ]
                        if day_idx < len(optimized_path.camp_indices)
                        else None
                    ),
                )
                for day_idx, segment in enumerate(optimized_path.get_day_segments())
            ],
            total_distance=float(
                sum(
                    dist_matrix[optimized_path.points[i]][optimized_path.points[i + 1]]
                    for i in range(len(optimized_path.points) - 1)
                )
            ),
            camping_spots=[
                all_points[optimized_path.points[i]]
                for i in optimized_path.camp_indices
            ],
            trail_network=[GeoJSONLineString(**t) for t in trail_network],
            points_of_interest=[
                all_points[i] for i in optimized_path.points if i < len(matched_pos)
            ],
        )

        return response

    except Exception as e:
        print(f"Error in generate_hiking_path: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_bounds(request, buffer_miles=5.0):
    """Calculate bounding box with buffer around points"""

    # Convert miles to degrees (approximate at Isle Royale's latitude)
    lat = request.start_point[0]  # latitude in degrees
    miles_per_lat_degree = 69.0  # constant
    miles_per_lon_degree = miles_per_lat_degree * np.cos(np.radians(lat))

    # Convert buffer to degrees
    lat_buffer = buffer_miles / miles_per_lat_degree
    lon_buffer = buffer_miles / miles_per_lon_degree

    points = [request.start_point]
    if request.end_point:
        points.append(request.end_point)

    # Calculate bounds
    min_lat = max(min(p[0] for p in points) - lat_buffer, -90)
    max_lat = min(max(p[0] for p in points) + lat_buffer, 90)
    min_lon = max(min(p[1] for p in points) - lon_buffer, -180)
    max_lon = min(max(p[1] for p in points) + lon_buffer, 180)

    return (min_lat, min_lon, max_lat, max_lon)
