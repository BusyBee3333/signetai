/**
 * Community detection for the entity graph.
 * Stub — full implementation in DP-5.
 */

import type { WriteDb } from "../db-accessor";

export interface ClusterResult {
	readonly communities: number;
	readonly modularity: number;
}

export function clusterEntities(_db: WriteDb, _agentId: string): ClusterResult {
	return { communities: 0, modularity: 0 };
}
