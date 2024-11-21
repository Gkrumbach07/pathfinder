'use server'

import { PathRequest, PathResponse } from "@/types/api";

export async function generatePath(params: PathRequest): Promise<PathResponse> {
	try {
		const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/generate-path`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(params),
		});

		if (!response.ok) {
			const error = await response.json();
			throw new Error(error.detail || 'Failed to generate path');
		}

		return await response.json();
	} catch (error) {
		console.error('Error generating path:', error);
		throw error;
	}
} 