import osmnx as ox
import overpy
from shapely.geometry import Point, LineString
import geopandas as gpd
import pandas as pd


class OSMFetcher:
    def __init__(self):
        self.api = overpy.Overpass()

    def get_data_for_bounds(self, bounds):
        """Get all required data for bounds"""
        query = f"""
        [out:json][timeout:25];
        (
            way["highway"]
                ({bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]});
        );
        out body;
        >;
        out skel qt;
        """

        try:
            print("Executing OSM query...")
            result = self.api.query(query)
            print(f"Query result: {len(result.ways)} ways found")

            # Convert to GeoDataFrame
            paths = []
            for way in result.ways:
                coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
                if (
                    len(coords) >= 2
                ):  # Ensure we have at least 2 points for a LineString
                    paths.append(
                        {
                            "geometry": LineString(coords),
                            "osm_id": way.id,
                            "highway_type": way.tags.get("highway", "unknown"),
                        }
                    )

            if not paths:
                print("No paths found")
                return (
                    self.create_empty_trails_gdf(),
                    self.create_empty_pois_gdf(),
                    self.create_empty_camps_gdf(),
                )

            # Create GeoDataFrame with explicit CRS
            trails_gdf = gpd.GeoDataFrame(
                paths, crs="EPSG:4326"  # WGS84 - standard lat/lon
            )

            # Print summary of highway types
            highway_counts = trails_gdf["highway_type"].value_counts()
            print("\nHighway types found:")
            for highway_type, count in highway_counts.items():
                print(f"- {highway_type}: {count}")

            print(f"Found {len(trails_gdf)} trail segments")

            # Fetch POIs and campsites
            pois_gdf = self.get_points_of_interest(bounds)
            camps_gdf = self.get_campsites(bounds)

            # Verify we have at least some data
            if trails_gdf.empty:
                raise ValueError("No hiking trails found in the specified area")
            if camps_gdf.empty:
                raise ValueError("No campsites found in the specified area")

            return trails_gdf, pois_gdf, camps_gdf

        except Exception as e:
            print(f"Error fetching trails: {str(e)}")
            print(f"Query was: {query}")
            return (
                self.create_empty_trails_gdf(),
                self.create_empty_pois_gdf(),
                self.create_empty_camps_gdf(),
            )

    def create_empty_trails_gdf(self):
        """Create an empty GeoDataFrame with correct schema for trails"""
        return gpd.GeoDataFrame(
            pd.DataFrame(columns=["geometry", "osm_id", "tags"]),
            geometry="geometry",
            crs="EPSG:4326",
        )

    def create_empty_pois_gdf(self):
        """Create an empty GeoDataFrame with correct schema for POIs"""
        return gpd.GeoDataFrame(
            pd.DataFrame(columns=["geometry", "osm_id", "name", "type", "tags"]),
            geometry="geometry",
            crs="EPSG:4326",
        )

    def create_empty_camps_gdf(self):
        """Create an empty GeoDataFrame with correct schema for campsites"""
        return gpd.GeoDataFrame(
            pd.DataFrame(columns=["geometry", "osm_id", "name", "tags"]),
            geometry="geometry",
            crs="EPSG:4326",
        )

    def get_hiking_trails(self, bounds):
        """Fetch all highways within bounds
        bounds: tuple of (min_lat, min_lon, max_lat, max_lon)
        """
        print(f"\nQuerying OSM for trails...")
        print(f"Bounds (south,west,north,east): {bounds}")
        print(
            f"Area size: {abs(bounds[2]-bounds[0]):.4f}° lat x {abs(bounds[3]-bounds[1]):.4f}° lon"
        )

        query = f"""
        [out:json][timeout:25];
        (
            // Get all highways
            way["highway"]
                ({bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]});
        );
        out body;
        >;
        out skel qt;
        """
        try:
            print(f"Executing OSM query...")
            result = self.api.query(query)
            print(f"Query result: {len(result.ways)} ways found")
        except Exception as e:
            print(f"Error fetching trails: {str(e)}")
            print(f"Query was: {query}")
            return self.create_empty_trails_gdf()

        paths = []
        for way in result.ways:
            coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
            if len(coords) >= 2:  # Ensure we have at least 2 points for a LineString
                try:
                    paths.append(
                        {
                            "geometry": LineString(coords),
                            "osm_id": way.id,
                            "tags": way.tags,
                            "highway_type": way.tags.get("highway", "unknown"),
                        }
                    )
                except Exception as e:
                    print(f"Error creating LineString: {str(e)}")
                    continue

        if not paths:
            print(f"No trails found in bounds: {bounds}")
            return self.create_empty_trails_gdf()

        print(f"Found {len(paths)} trail segments")

        try:
            # Create GeoDataFrame with explicit schema
            df = pd.DataFrame(paths)
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

            # Print summary of highway types found
            highway_counts = gdf["highway_type"].value_counts()
            print("\nHighway types found:")
            for highway_type, count in highway_counts.items():
                print(f"- {highway_type}: {count}")

            return gdf

        except Exception as e:
            print(f"Error creating GeoDataFrame: {str(e)}")
            print("Paths data:", paths[:5])  # Print first 5 paths for debugging
            return self.create_empty_trails_gdf()

    def get_points_of_interest(self, bounds):
        """Fetch POIs within bounds"""
        query = f"""
        [out:json][timeout:25];
        (
            node["tourism"~"viewpoint|information|picnic_site"]
                ({bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]});
            node["natural"~"peak|waterfall"]
                ({bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]});
        );
        out body;
        >;
        out skel qt;
        """
        try:
            result = self.api.query(query)
        except Exception as e:
            print(f"Error fetching POIs: {str(e)}")
            return self.create_empty_pois_gdf()

        pois = []
        for node in result.nodes:
            pois.append(
                {
                    "geometry": Point(float(node.lon), float(node.lat)),
                    "osm_id": node.id,
                    "name": node.tags.get("name", "Unknown"),
                    "type": node.tags.get("tourism") or node.tags.get("natural"),
                    "tags": node.tags,
                }
            )

        if not pois:
            return self.create_empty_pois_gdf()

        return gpd.GeoDataFrame(pois, geometry="geometry", crs="EPSG:4326")

    def get_campsites(self, bounds):
        """Fetch campsites within bounds"""
        query = f"""
        [out:json][timeout:25];
        (
            node["tourism"="camp_site"]
                ({bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]});
            node["leisure"="camp_site"]
                ({bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]});
        );
        out body;
        >;
        out skel qt;
        """
        try:
            result = self.api.query(query)
        except Exception as e:
            print(f"Error fetching campsites: {str(e)}")
            return self.create_empty_camps_gdf()

        campsites = []
        for node in result.nodes:
            campsites.append(
                {
                    "geometry": Point(float(node.lon), float(node.lat)),
                    "osm_id": node.id,
                    "name": node.tags.get("name", "Unknown Campsite"),
                    "tags": node.tags,
                }
            )

        if not campsites:
            return self.create_empty_camps_gdf()

        return gpd.GeoDataFrame(campsites, geometry="geometry", crs="EPSG:4326")
