'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import PathControls from '@/components/PathControls';
import { generatePath } from './actions';
import { PathRequest, PathResponse, PathPoint, GeoJSONLineString } from '@/types/api';
import { useToast } from '@/hooks/use-toast';

// Dynamically import HikingMap with no SSR
const HikingMap = dynamic(() => import('@/components/HikingMap'), { ssr: false });

export default function Home() {
	const [startPoint, setStartPoint] = useState<[number, number]>();
	const [pathData, setPathData] = useState<PathResponse>();
	const [isLoading, setIsLoading] = useState(false);
	const [highlightedPoint, setHighlightedPoint] = useState<PathPoint>();
	const [highlightedSegment, setHighlightedSegment] = useState<GeoJSONLineString>();
	const { toast } = useToast();

	const handleMapClick = (coords: [number, number]) => {
		setStartPoint(coords);
	};

	const handleGeneratePath = async (params: PathRequest) => {
		try {
			setIsLoading(true);
			const data = await generatePath(params);
			setPathData(data);
			console.log(data);

			if (data?.daily_segments?.[0]) {
				toast({
					title: "Path Generated",
					description: `Total distance: ${data.total_distance.toFixed(2)} miles`,
				});
			}
		} catch (error) {
			toast({
				title: "Error",
				description: error instanceof Error ? error.message : "Failed to generate path",
				variant: "destructive",
			});
		} finally {
			setIsLoading(false);
		}
	};

	const handleDaySelect = (segment?: GeoJSONLineString) => {
		setHighlightedSegment(segment);
	};

	return (
		<div className="flex h-screen bg-background">
			<div className="w-1/4 p-4 border-r overflow-auto">
				<PathControls
					onGeneratePath={handleGeneratePath}
					startPoint={startPoint}
					isLoading={isLoading}
					pathData={pathData}
					onPointSelect={setHighlightedPoint}
					onDaySelect={handleDaySelect} />
			</div>
			<div className="w-3/4 p-4">
				<HikingMap
					startPoint={startPoint}
					path={pathData?.path}
					daily_segments={pathData?.daily_segments.map(s => s.path)}
					trailNetwork={pathData?.trail_network}
					highlightedPoint={highlightedPoint}
					highlightedSegment={highlightedSegment}
					points_of_interest={pathData?.points_of_interest}
					camping_spots={pathData?.camping_spots}
					onMapClick={handleMapClick}
				/>
			</div>
		</div>
	);
} 