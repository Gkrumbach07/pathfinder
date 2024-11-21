'use client';

import { useState, ChangeEvent } from 'react';
import { PathType } from '../types';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
	Accordion,
	AccordionContent,
	AccordionItem,
	AccordionTrigger,
} from "@/components/ui/accordion";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Loader2, Tent, Mountain } from "lucide-react";
import { PathRequest, PathResponse, PathPoint, GeoJSONLineString } from '@/types/api';

interface PathControlsProps {
	onGeneratePath: (params: PathRequest) => void;
	startPoint?: [number, number];
	endPoint?: [number, number];
	isLoading: boolean;
	pathData?: PathResponse;
	onPointSelect?: (point: PathPoint) => void;
	onDaySelect?: (segment: GeoJSONLineString | undefined) => void;
}

const PathControls = ({
	onGeneratePath,
	startPoint,
	endPoint,
	isLoading,
	pathData,
	onPointSelect,
	onDaySelect,
}: PathControlsProps) => {
	const [selectedDay, setSelectedDay] = useState<number | null>(null);
	const [nights, setNights] = useState(2);
	const [maxDayDistance, setMaxDayDistance] = useState(12);
	const [pathType, setPathType] = useState<PathType>(PathType.LOOP);


	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		if (!startPoint) return;

		onGeneratePath({
			path_type: pathType,
			start_point: startPoint,
			nights,
			max_day_distance: maxDayDistance
		});
	};

	const handleNumberChange = (e: ChangeEvent<HTMLInputElement>, setter: (value: number) => void) => {
		const value = parseInt(e.target.value);
		if (!isNaN(value)) {
			setter(value);
		}
	};

	const handleDayClick = (dayIndex: number) => {
		if (selectedDay === dayIndex) {
			setSelectedDay(null);
			onDaySelect?.(undefined);
		} else {
			setSelectedDay(dayIndex);
			if (pathData?.daily_segments[dayIndex]) {
				onDaySelect?.(pathData.daily_segments[dayIndex].path);
			}
		}
	};

	return (
		<div className="space-y-4">
			<Card>
				<CardHeader>
					<CardTitle>Path Settings</CardTitle>
				</CardHeader>
				<CardContent>
					<form onSubmit={handleSubmit} className="space-y-4">
						<div className="space-y-2">
							<Label>Path Type</Label>
							<Select
								value={pathType}
								onValueChange={(value: PathType) => setPathType(value)}
							>
								<SelectTrigger>
									<SelectValue placeholder="Select path type" />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value={PathType.LOOP}>Loop</SelectItem>
									<SelectItem value={PathType.POINT_TO_POINT}>Point to Point</SelectItem>
									<SelectItem value={PathType.OPEN_ENDED}>Open Ended</SelectItem>
								</SelectContent>
							</Select>
						</div>

						<div className="space-y-2">
							<Label>
								Start Point: {startPoint ? `${startPoint[0].toFixed(6)}, ${startPoint[1].toFixed(6)}` : 'Not set'}
							</Label>
						</div>

						{pathType === PathType.POINT_TO_POINT && (
							<div className="space-y-2">
								<Label>
									End Point: {endPoint ? `${endPoint[0].toFixed(6)}, ${endPoint[1].toFixed(6)}` : 'Not set'}
								</Label>
							</div>
						)}

						<div className="space-y-2">
							<Label>Number of Nights</Label>
							<Input
								type="number"
								value={nights}
								onChange={(e) => handleNumberChange(e, setNights)}
								min={0}
							/>
						</div>

						<div className="space-y-2">
							<Label>Max Distance per Day (miles)</Label>
							<Input
								type="number"
								value={maxDayDistance}
								onChange={(e) => handleNumberChange(e, setMaxDayDistance)}
								min={1}
							/>
						</div>

						<Button
							type="submit"
							disabled={!startPoint || (pathType === PathType.POINT_TO_POINT && !endPoint) || isLoading}
							className="w-full"
						>
							{isLoading ? (
								<>
									<Loader2 className="mr-2 h-4 w-4 animate-spin" />
									Generating...
								</>
							) : (
								'Generate Path'
							)}
						</Button>
					</form>
				</CardContent>
			</Card>

			{pathData && (
				<>
					<Card>
						<CardHeader>
							<CardTitle>Overview</CardTitle>
						</CardHeader>
						<CardContent className="space-y-2">
							<div className="flex justify-between">
								<span>Total Distance:</span>
								<span>{pathData.total_distance.toFixed(2)} miles</span>
							</div>
							<div className="flex justify-between">
								<span>Days of Hiking:</span>
								<span>{pathData.daily_segments.length}</span>
							</div>
							<div className="flex justify-between">
								<span>Nights Camping:</span>
								<span>{pathData.camping_spots.length}</span>
							</div>
							<div className="flex justify-between">
								<span>Points of Interest:</span>
								<span>{pathData.points_of_interest.length}</span>
							</div>
						</CardContent>
					</Card>

					<Card>
						<CardHeader>
							<CardTitle>Daily Segments</CardTitle>
						</CardHeader>
						<CardContent>
							<div className="space-y-4">
								{pathData.daily_segments.map((segment, idx) => (
									<div
										key={idx}
										className={`space-y-2 p-2 rounded-lg cursor-pointer transition-colors ${selectedDay === idx ? 'bg-secondary' : 'hover:bg-secondary/50'
											}`}
										onClick={() => handleDayClick(idx)}
									>
										<h4 className="font-medium">Day {idx + 1}</h4>

										{segment.points.map((point, pointIdx) => (
											<div key={pointIdx}>
												<div
													className="pl-4 space-y-1 hover:bg-accent p-1 rounded cursor-pointer"
													onClick={(e) => {
														e.stopPropagation();
														onPointSelect?.(point);
													}}
												>
													<div className="flex items-center">
														{pathData.camping_spots.includes(point) ? (
															<Tent className="h-4 w-4 mr-2" />
														) : (
															<Mountain className="h-4 w-4 mr-2" />
														)}
														<span>{point.name || 'Trail Point'}</span>
														{point.type && (
															<span className="ml-2 text-sm text-muted-foreground">
																({point.type})
															</span>
														)}
													</div>
												</div>
												{pointIdx < segment.distances.length && (
													<div className="text-sm text-muted-foreground pl-6">
														↓ {segment.distances[pointIdx].distance.toFixed(2)} miles
													</div>
												)}
											</div>
										))}

										<Badge variant="outline">
											Total: {segment.total_distance.toFixed(2)} miles
										</Badge>
									</div>
								))}
							</div>
						</CardContent>
					</Card>

					<Accordion type="single" collapsible className="w-full">
						<AccordionItem value="camping">
							<AccordionTrigger>Camping Spots</AccordionTrigger>
							<AccordionContent>
								<div className="space-y-2">
									{pathData.camping_spots.map((spot, idx) => (
										<div
											key={idx}
											className="flex items-center hover:bg-secondary p-2 rounded cursor-pointer"
											onClick={() => onPointSelect?.(spot)}
										>
											<Tent className="h-4 w-4 mr-2" />
											<span>Night {idx + 1}: {spot.name || 'Unnamed Spot'}</span>
										</div>
									))}
								</div>
							</AccordionContent>
						</AccordionItem>

						<AccordionItem value="poi">
							<AccordionTrigger>Points of Interest</AccordionTrigger>
							<AccordionContent>
								<div className="space-y-2">
									{pathData.points_of_interest.map((poi, idx) => (
										<div
											key={idx}
											className="flex items-center hover:bg-secondary p-2 rounded cursor-pointer"
											onClick={() => onPointSelect?.(poi)}
										>
											<Mountain className="h-4 w-4 mr-2" />
											<span>{poi.name || 'Unnamed POI'}</span>
											{poi.type && (
												<span className="ml-2 text-sm text-muted-foreground">
													({poi.type})
												</span>
											)}
										</div>
									))}
								</div>
							</AccordionContent>
						</AccordionItem>
					</Accordion>
				</>
			)}
		</div>
	);
};

export default PathControls; 