from pydantic import BaseModel
from typing import List, Optional, Tuple
from enum import Enum


class PathType(str, Enum):
    LOOP = "loop"
    POINT_TO_POINT = "point_to_point"
    OPEN_ENDED = "open_ended"


class PathRequest(BaseModel):
    path_type: PathType
    start_point: Tuple[float, float]  # [lat, lon]
    end_point: Optional[Tuple[float, float]] = None
    nights: int = 2
    max_day_distance: float = 12.0  # miles


class PathPoint(BaseModel):
    point: Tuple[float, float]
    node_idx: int
    name: Optional[str] = None
    type: Optional[str] = None
    osm_id: Optional[int] = None


class GeoJSONLineString(BaseModel):
    type: str = "LineString"
    coordinates: List[Tuple[float, float]]


class SegmentDistance(BaseModel):
    from_point: PathPoint
    to_point: PathPoint
    distance: float


class DailySegment(BaseModel):
    path: GeoJSONLineString
    points: List[PathPoint]  # All points visited this day in order
    distances: List[SegmentDistance]  # Distances between consecutive points
    total_distance: float
    camping_spot: Optional[PathPoint]  # The camping spot for this day, if any


class PathResponse(BaseModel):
    path: GeoJSONLineString
    daily_segments: List[DailySegment]
    total_distance: float
    camping_spots: List[PathPoint]
    trail_network: List[GeoJSONLineString]
    points_of_interest: List[PathPoint]
