import { PathType } from './index';

export interface PathRequest {
	path_type: PathType;
	start_point: [number, number];
	end_point?: [number, number];
	nights: number;
	max_day_distance: number;
}

export interface PathPoint {
	point: [number, number];
	node_idx: number;
	name?: string;
	type?: string;
	osm_id?: number;
}

export interface GeoJSONLineString {
	type: "LineString";
	coordinates: [number, number][];
}

export interface SegmentDistance {
	from_point: PathPoint;
	to_point: PathPoint;
	distance: number;
}

export interface DailySegment {
	path: GeoJSONLineString;
	points: PathPoint[];
	distances: SegmentDistance[];
	total_distance: number;
	camping_spot?: PathPoint;
}

export interface PathResponse {
	path: GeoJSONLineString;
	daily_segments: DailySegment[];
	score: number;
	total_distance: number;
	camping_spots: PathPoint[];
	trail_network: GeoJSONLineString[];
	points_of_interest: PathPoint[];
}