'use client';

import { useEffect, useRef } from 'react';
import { Card } from "@/components/ui/card";
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import { fromLonLat, transform } from 'ol/proj';
import { Feature } from 'ol';
import { Point, LineString } from 'ol/geom';
import { Style, Stroke, Circle, Fill, Text, RegularShape, Icon } from 'ol/style';
import { MapBrowserEvent } from 'ol';
import { PathPoint, GeoJSONLineString } from '@/types/api';
import { CampsiteIcon, PeakIcon, ViewpointIcon } from './icons';

interface HikingMapProps {
	startPoint?: [number, number];
	path?: GeoJSONLineString;
	daily_segments?: GeoJSONLineString[];
	trailNetwork?: GeoJSONLineString[];
	highlightedPoint?: PathPoint;
	highlightedSegment?: GeoJSONLineString;
	points_of_interest?: PathPoint[];
	camping_spots?: PathPoint[];
	onMapClick: (coords: [number, number]) => void;
}

const getIconForType = (type: string | null | undefined) => {
	switch (type) {
		case 'peak':
			return PeakIcon;
		case 'viewpoint':
			return ViewpointIcon;
		default:
			return null;
	}
};

const HikingMap = ({ startPoint, path, daily_segments, trailNetwork, highlightedPoint, highlightedSegment, points_of_interest, camping_spots, onMapClick }: HikingMapProps) => {
	const mapRef = useRef<HTMLDivElement>(null);
	const mapInstanceRef = useRef<Map | null>(null);
	const vectorLayerRef = useRef<VectorLayer<VectorSource>>();

	// Initialize map only once
	useEffect(() => {
		if (typeof window === 'undefined' || !mapRef.current || mapInstanceRef.current) return;

		// Initialize map
		const map = new Map({
			target: mapRef.current,
			layers: [
				new TileLayer({
					source: new OSM()
				})
			],
			view: new View({
				center: fromLonLat([-89.155861, 47.916305]),
				zoom: 12
			})
		});

		// Create vector layer
		const vectorLayer = new VectorLayer({
			source: new VectorSource(),
			zIndex: 1
		});
		map.addLayer(vectorLayer);

		mapInstanceRef.current = map;
		vectorLayerRef.current = vectorLayer;

		// Add click handler
		map.on('click', (event: MapBrowserEvent<MouseEvent>) => {
			const coords = event.coordinate;
			const [lon, lat] = transform(coords, 'EPSG:3857', 'EPSG:4326');
			onMapClick([lat, lon]);
		});

		return () => {
			if (mapInstanceRef.current) {
				mapInstanceRef.current.setTarget(undefined);
				mapInstanceRef.current = null;
			}
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	// Update markers and path when props change
	useEffect(() => {
		if (!mapInstanceRef.current) return;

		const map = mapInstanceRef.current;

		// Clear existing vector layers
		map.getLayers().getArray()
			.filter((layer): layer is VectorLayer<VectorSource> => layer instanceof VectorLayer)
			.forEach((layer) => map.removeLayer(layer));

		// Create separate sources for different layer types
		const baseSource = new VectorSource();  // For trail network
		const pathSource = new VectorSource();  // For normal path segments
		const highlightSource = new VectorSource();  // For highlighted segments
		const markerSource = new VectorSource();  // For points

		// Add start point marker
		if (startPoint) {
			const startFeature = new Feature({
				geometry: new Point(fromLonLat([startPoint[1], startPoint[0]]))  // Convert [lat,lon] to [lon,lat]
			});
			startFeature.setStyle(new Style({
				image: new Circle({
					radius: 8,
					fill: new Fill({ color: 'green' }),
					stroke: new Stroke({
						color: 'white',
						width: 2
					})
				}),
				text: new Text({
					text: 'Start',
					offsetY: -15,
					fill: new Fill({ color: 'black' }),
					stroke: new Stroke({ color: 'white', width: 2 }),
					font: '12px sans-serif'
				})
			}));
			markerSource.addFeature(startFeature);
		}

		// Add trail network (bottom layer)
		if (trailNetwork) {
			trailNetwork.forEach(line => {
				const pathFeature = new Feature({
					geometry: new LineString(
						line.coordinates.map(coord => fromLonLat(coord))
					)
				});
				pathFeature.setStyle(new Style({
					stroke: new Stroke({
						color: 'rgba(128, 128, 128, 0.5)',
						width: 1
					})
				}));
				baseSource.addFeature(pathFeature);
			});
		}

		// Add main path (middle layer)
		if (path) {
			const pathFeature = new Feature({
				geometry: new LineString(
					path.coordinates.map(coord => fromLonLat(coord))
				)
			});
			pathFeature.setStyle(new Style({
				stroke: new Stroke({
					color: 'blue',
					width: 3
				})
			}));
			pathSource.addFeature(pathFeature);
		}

		// Add highlighted segment (top layer)
		if (highlightedSegment) {
			const coords = highlightedSegment.coordinates.map(coord => fromLonLat(coord));
			const pathFeature = new Feature({
				geometry: new LineString(coords)
			});

			const styles = [
				new Style({
					stroke: new Stroke({
						color: 'red',
						width: 4
					})
				})
			];

			// Add arrows every n points
			const n = Math.max(1, Math.floor(coords.length / 8));  // Adjusted spacing
			for (let i = n; i < coords.length - n; i += n) {
				const start = coords[i - 1];
				const end = coords[i];
				const dx = end[0] - start[0];
				const dy = end[1] - start[1];
				const rotation = Math.atan2(dy, dx);

				styles.push(new Style({
					geometry: new Point(coords[i]),
					image: new RegularShape({
						points: 3,  // Triangle
						radius: 10,  // Slightly larger
						rotation: -rotation + Math.PI / 2,  // Adjusted rotation
						fill: new Fill({ color: 'red' }),
						stroke: new Stroke({ color: 'white', width: 2 })
					})
				}));
			}

			pathFeature.setStyle(styles);
			highlightSource.addFeature(pathFeature);
		}

		// Add markers for POIs
		if (points_of_interest) {
			points_of_interest.forEach((poi) => {
				const isHighlighted = highlightedPoint?.point[0] === poi.point[0] &&
					highlightedPoint?.point[1] === poi.point[1];

				const markerFeature = new Feature({
					geometry: new Point(fromLonLat([poi.point[0], poi.point[1]]))
				});

				const iconSvg = getIconForType(poi.type);
				markerFeature.setStyle(new Style({
					image: iconSvg ? new Icon({
						src: 'data:image/svg+xml;utf8,' + encodeURIComponent(iconSvg),
						scale: isHighlighted ? 1.5 : 1,
						color: '#000000',
						opacity: 1
					}) : new Circle({
						radius: isHighlighted ? 8 : 6,
						fill: new Fill({ color: '#FFD700' }),
						stroke: new Stroke({
							color: 'white',
							width: isHighlighted ? 2 : 1
						})
					}),
					text: new Text({
						text: poi.name || 'POI',
						offsetY: -20,
						fill: new Fill({ color: 'black' }),
						stroke: new Stroke({ color: 'white', width: 2 }),
						font: '12px sans-serif'
					})
				}));
				markerSource.addFeature(markerFeature);
			});
		}

		// Add markers for campsites
		if (camping_spots) {
			camping_spots.forEach((camp) => {
				const isHighlighted = highlightedPoint?.point[0] === camp.point[0] &&
					highlightedPoint?.point[1] === camp.point[1];

				const markerFeature = new Feature({
					geometry: new Point(fromLonLat([camp.point[0], camp.point[1]]))
				});

				markerFeature.setStyle(new Style({
					image: new Icon({
						src: 'data:image/svg+xml;utf8,' + encodeURIComponent(CampsiteIcon),
						scale: isHighlighted ? 1.5 : 1,
						color: '#FF0000',
						opacity: 1
					}),
					text: new Text({
						text: camp.name || 'Camp',
						offsetY: -20,
						fill: new Fill({ color: 'black' }),
						stroke: new Stroke({ color: 'white', width: 2 }),
						font: '12px sans-serif'
					})
				}));
				markerSource.addFeature(markerFeature);
			});
		}

		// Add layers in correct order
		map.addLayer(new VectorLayer({ source: baseSource, zIndex: 1 }));
		map.addLayer(new VectorLayer({ source: pathSource, zIndex: 2 }));
		map.addLayer(new VectorLayer({ source: highlightSource, zIndex: 3 }));
		map.addLayer(new VectorLayer({ source: markerSource, zIndex: 4 }));

	}, [startPoint, path, daily_segments, trailNetwork, highlightedPoint, highlightedSegment, points_of_interest, camping_spots]);

	return (
		<Card className="w-full h-full">
			<div ref={mapRef} className="w-full h-full" />
		</Card>
	);
};

export default HikingMap; 