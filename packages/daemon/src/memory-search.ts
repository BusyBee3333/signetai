/**
 * Hybrid recall search orchestration.
 *
 * Extracted from daemon.ts — this module contains the pure search logic
 * between "parse request" and "format response". The route handler in
 * daemon.ts is now a thin HTTP wrapper that delegates here.
 */

import { vectorSearch } from "@signet/core";
import { getDbAccessor } from "./db-accessor";
import { logger } from "./logger";
import type { EmbeddingConfig, MemorySearchConfig, ResolvedMemoryConfig } from "./memory-config";
import { getGraphBoostIds, tokenizeGraphQuery } from "./pipeline/graph-search";
import { FTS_STOP } from "./pipeline/stop-words";
import {
	resolveFocalEntities,
	setTraversalStatus,
	traverseKnowledgeGraph,
} from "./pipeline/graph-traversal";
import { constructContextBlocks } from "./pipeline/context-construction";
import { type RerankCandidate, noopReranker, rerank } from "./pipeline/reranker";
import { createEmbeddingReranker } from "./pipeline/reranker-embedding";

// ---------------------------------------------------------------------------
// Public interfaces
// ---------------------------------------------------------------------------

export interface RecallParams {
	query: string;
	keywordQuery?: string;
	limit?: number;
	agentId?: string;
	type?: string;
	tags?: string;
	who?: string;
	pinned?: boolean;
	importance_min?: number;
	since?: string;
	until?: string;
	scope?: string | null;
}

export interface RecallResult {
	id: string;
	content: string;
	content_length: number;
	truncated: boolean;
	score: number;
	source: string;
	type: string;
	tags: string | null;
	pinned: boolean;
	importance: number;
	who: string;
	project: string | null;
	created_at: string;
	supplementary?: boolean;
}

export interface RecallResponse {
	results: RecallResult[];
	query: string;
	method: "hybrid" | "keyword";
	entities?: Array<{
		name: string;
		type: string;
		aspects: Array<{
			name: string;
			attributes: Array<{ content: string; status: string; importance: number }>;
		}>;
	}>;
}

export type EmbedFn = (text: string, cfg: EmbeddingConfig) => Promise<number[] | null>;

// ---------------------------------------------------------------------------
// Filter clause builder (private)
// ---------------------------------------------------------------------------

interface FilterClause {
	sql: string;
	args: unknown[];
}

function buildFilterClause(params: RecallParams): FilterClause {
	const parts: string[] = [];
	const args: unknown[] = [];

	// Scope isolation: explicit scope filters to that scope, undefined
	// defaults to excluding all scoped memories from normal searches.
	if (params.scope !== undefined) {
		if (params.scope === null) {
			parts.push("m.scope IS NULL");
		} else {
			parts.push("m.scope = ?");
			args.push(params.scope);
		}
	} else {
		parts.push("m.scope IS NULL");
	}

	if (params.type) {
		parts.push("m.type = ?");
		args.push(params.type);
	}
	if (params.tags) {
		for (const t of params.tags
			.split(",")
			.map((s) => s.trim())
			.filter(Boolean)) {
			parts.push("m.tags LIKE ?");
			args.push(`%${t}%`);
		}
	}
	if (params.who) {
		parts.push("m.who = ?");
		args.push(params.who);
	}
	if (params.pinned) {
		parts.push("m.pinned = 1");
	}
	if (typeof params.importance_min === "number") {
		parts.push("m.importance >= ?");
		args.push(params.importance_min);
	}
	if (params.since) {
		parts.push("m.created_at >= ?");
		args.push(params.since);
	}
	if (params.until) {
		parts.push("m.created_at <= ?");
		args.push(params.until);
	}

	return {
		sql: parts.length ? ` AND ${parts.join(" AND ")}` : "",
		args,
	};
}

// ---------------------------------------------------------------------------
// FTS5 query sanitization
// ---------------------------------------------------------------------------

/**
 * Sanitize a query string for FTS5 MATCH.
 *
 * Strips FTS5 syntax characters, removes stop words, and quotes each
 * token as a literal. Short queries (<=3 tokens) use implicit AND for
 * precision; longer queries use OR so BM25 IDF ranks by term importance.
 */
function sanitizeFtsQuery(raw: string): string {
	const tokens = raw
		.replace(/'/g, " ")
		.split(/\s+/)
		.map((token) => {
			const cleaned = token.replace(/[":()^*?]/g, "").trim().toLowerCase();
			if (!cleaned || cleaned.length < 2) return null;
			if (FTS_STOP.has(cleaned)) return null;
			return `"${cleaned}"`;
		})
		.filter(Boolean) as string[];

	if (tokens.length === 0) return "";
	// Short queries (<=3 content tokens): implicit AND for precision.
	// Longer queries: OR so BM25 IDF ranks by term importance.
	if (tokens.length <= 3) return tokens.join(" ");
	return tokens.join(" OR ");
}

// ---------------------------------------------------------------------------
// Rehearsal boost (shared between traversal-primary and legacy paths)
// ---------------------------------------------------------------------------

function applyRehearsalBoost(
	scored: Array<{ id: string; score: number; source: string }>,
	search: MemorySearchConfig,
): void {
	if (!search.rehearsal_enabled || search.rehearsal_weight <= 0 || scored.length === 0) return;
	try {
		const ids = scored.map((s) => s.id);
		const placeholders = ids.map(() => "?").join(", ");
		const accessRows = getDbAccessor().withReadDb(
			(db) =>
				db
					.prepare(
						`SELECT id, access_count, last_accessed
						 FROM memories
						 WHERE id IN (${placeholders})`,
					)
					.all(...ids) as Array<{
					id: string;
					access_count: number;
					last_accessed: string | null;
				}>,
		);

		const nowMs = Date.now();
		const rw = search.rehearsal_weight;
		const accessMap = new Map(accessRows.map((r) => [r.id, r]));
		for (const s of scored) {
			const row = accessMap.get(s.id);
			if (!row || row.access_count <= 0) continue;
			const daysSinceAccess = row.last_accessed
				? (nowMs - new Date(row.last_accessed).getTime()) / 86_400_000
				: search.rehearsal_half_life_days;
			const recencyFactor = 0.5 ** (daysSinceAccess / search.rehearsal_half_life_days);
			const boost = rw * Math.log(row.access_count + 1) * recencyFactor;
			s.score *= 1 + boost;
		}
		scored.sort((a, b) => b.score - a.score);
	} catch (e) {
		logger.warn("memory", "Rehearsal boost failed (non-fatal)", {
			error: e instanceof Error ? e.message : String(e),
		});
	}
}

// ---------------------------------------------------------------------------
// Main search orchestration
// ---------------------------------------------------------------------------

export async function hybridRecall(
	params: RecallParams,
	cfg: ResolvedMemoryConfig,
	embedFn: EmbedFn,
): Promise<RecallResponse> {
	const query = params.query;
	const keywordQuery = sanitizeFtsQuery((params.keywordQuery ?? params.query).trim());
	const limit = params.limit ?? 10;
	const alpha = cfg.search.alpha;
	const minScore = cfg.search.min_score;

	const filter = buildFilterClause(params);
	const scoped = params.scope !== undefined;

	// --- BM25 keyword search via FTS5 ---
	const bm25Map = new Map<string, number>();
	try {
		getDbAccessor().withReadDb((db) => {
			const ftsRows = (
				db.prepare(`
        SELECT m.id, bm25(memories_fts) AS raw_score
        FROM memories_fts
        JOIN memories m ON memories_fts.rowid = m.rowid
        WHERE memories_fts MATCH ?${filter.sql}
        ORDER BY raw_score
        LIMIT ?
      `) as any
			).all(keywordQuery, ...filter.args, cfg.search.top_k) as Array<{
				id: string;
				raw_score: number;
			}>;

			// Min-max normalize BM25 scores to [0,1] within the batch
			const rawScores = ftsRows.map((r) => Math.abs(r.raw_score));
			const maxRaw =
				rawScores.length > 0 ? Math.max(...rawScores) : 1;
			const normalizer = maxRaw > 0 ? maxRaw : 1;
			for (const row of ftsRows) {
				const normalised = Math.abs(row.raw_score) / normalizer;
				bm25Map.set(row.id, normalised);
			}
		});
	} catch (e) {
		logger.warn("memory", "FTS search failed, continuing with vector only", {
			error: e instanceof Error ? e.message : String(e),
		});
	}

	// --- Query embedding (used by reranker even when vector search is skipped) ---
	let queryVecF32: Float32Array | null = null;
	try {
		const queryVec = await embedFn(query, cfg.embedding);
		if (queryVec) queryVecF32 = new Float32Array(queryVec);
	} catch (e) {
		logger.warn("memory", "Embedding failed", { error: String(e) });
	}

	// --- Vector search via sqlite-vec ---
	// Skipped for scoped queries: sqlite-vec cannot pre-filter by scope,
	// so out-of-scope results dominate the ranked list and displace
	// in-scope FTS5/traversal matches at the pre-hydration truncation.
	// Graph traversal provides the structural retrieval path (DP-6).
	const vectorMap = new Map<string, number>();
	if (!scoped && queryVecF32) {
		try {
			getDbAccessor().withReadDb((db) => {
				const vecResults = vectorSearch(db as any, queryVecF32!, {
					limit: cfg.search.top_k,
					type: params.type as "fact" | "preference" | "decision" | undefined,
				});
				for (const r of vecResults) {
					vectorMap.set(r.id, r.score);
				}
			});
		} catch (e) {
			logger.warn("memory", "Vector search failed, using keyword only", {
				error: String(e),
			});
		}
	}

	// --- Flat search: merge BM25 + vector scores ---
	const allIds = new Set([...bm25Map.keys(), ...vectorMap.keys()]);
	const flatScored: Array<{ id: string; score: number; source: string }> = [];

	for (const id of allIds) {
		const bm25 = bm25Map.get(id) ?? 0;
		const vec = vectorMap.get(id) ?? 0;
		let score: number;
		let source: string;

		if (bm25 > 0 && vec > 0) {
			score = alpha * vec + (1 - alpha) * bm25;
			source = "hybrid";
		} else if (vec > 0) {
			score = vec;
			source = "vector";
		} else {
			score = bm25;
			source = "keyword";
		}

		if (score >= minScore) flatScored.push({ id, score, source });
	}

	flatScored.sort((a, b) => b.score - a.score);

	// --- Score pipeline: traversal-primary vs legacy boost ---
	const traversalPrimary = cfg.pipelineV2.graph.enabled
		&& cfg.pipelineV2.traversal?.enabled
		&& cfg.pipelineV2.traversal?.primary !== false;

	let scored: Array<{ id: string; score: number; source: string }>;

	if (traversalPrimary) {
		// Channel A: graph traversal (primary retrieval path per DP-6)
		const traversalScored: Array<{ id: string; score: number; source: string }> = [];

		if (cfg.pipelineV2.traversal) {
			try {
				const traversalCfg = cfg.pipelineV2.traversal;
				const queryTokens = tokenizeGraphQuery(query);
				if (queryTokens.length > 0) {
					const agentId = params.agentId ?? "default";
					const focal = getDbAccessor().withReadDb((db) =>
						resolveFocalEntities(db, agentId, { queryTokens }),
					);

					if (focal.entityIds.length > 0) {
						const traversal = getDbAccessor().withReadDb((db) =>
							traverseKnowledgeGraph(focal.entityIds, db, agentId, {
								maxAspectsPerEntity: traversalCfg.maxAspectsPerEntity,
								maxAttributesPerAspect: traversalCfg.maxAttributesPerAspect,
								maxDependencyHops: traversalCfg.maxDependencyHops,
								minDependencyStrength: traversalCfg.minDependencyStrength,
								maxBranching: traversalCfg.maxBranching,
								maxTraversalPaths: traversalCfg.maxTraversalPaths,
								minConfidence: traversalCfg.minConfidence,
								timeoutMs: traversalCfg.timeoutMs,
								scope: params.scope,
							}),
						);

						for (const [memoryId, importance] of traversal.memoryScores) {
							traversalScored.push({
								id: memoryId,
								score: Math.max(minScore, Math.min(1, importance)),
								source: "traversal",
							});
						}

						setTraversalStatus({
							phase: "recall",
							at: new Date().toISOString(),
							source: focal.source,
							focalEntityNames: focal.entityNames,
							focalEntities: focal.entityIds.length,
							traversedEntities: traversal.entityCount,
							memoryCount: traversal.memoryIds.size,
							constraintCount: traversal.constraints.length,
							timedOut: traversal.timedOut,
						});
					}
				}
			} catch (e) {
				logger.warn("memory", "Traversal channel failed (non-fatal)", {
					error: e instanceof Error ? e.message : String(e),
				});
			}
		}

		// Channel B merge: traversal memories first, flat fills remaining slots.
		// Cap gap-fill so OR fan-out doesn't flood the merge and dilute traversal.
		const traversalIds = new Set(traversalScored.map((s) => s.id));
		const gapBudget = Math.max(0, limit - traversalScored.length);
		const gapFill = flatScored.filter((s) => !traversalIds.has(s.id)).slice(0, gapBudget);
		scored = [...traversalScored, ...gapFill];
		scored.sort((a, b) => b.score - a.score);

		applyRehearsalBoost(scored, cfg.search);
	} else {
		scored = flatScored;

		applyRehearsalBoost(scored, cfg.search);

		// --- Graph boost: pull up memories linked via knowledge graph ---
		if (cfg.pipelineV2.graph.enabled && cfg.pipelineV2.graph.boostWeight > 0) {
			try {
				const graphResult = getDbAccessor().withReadDb((db) =>
					getGraphBoostIds(query, db, cfg.pipelineV2.graph.boostTimeoutMs, params.agentId),
				);
				if (graphResult.graphLinkedIds.size > 0) {
					const gw = cfg.pipelineV2.graph.boostWeight;
					for (const s of scored) {
						if (graphResult.graphLinkedIds.has(s.id)) {
							s.score = (1 - gw) * s.score + gw;
						}
					}
					scored.sort((a, b) => b.score - a.score);
				}
			} catch (e) {
				logger.warn("memory", "Graph boost failed (non-fatal)", {
					error: e instanceof Error ? e.message : String(e),
				});
			}
		}

		// --- KA traversal boost: structural one-hop retrieval via KA tables ---
		if (cfg.pipelineV2.graph.enabled && cfg.pipelineV2.traversal?.enabled) {
			try {
				const traversalCfg = cfg.pipelineV2.traversal;
				const queryTokens = tokenizeGraphQuery(query);
				if (queryTokens.length > 0) {
					const agentId = params.agentId ?? "default";
					const focal = getDbAccessor().withReadDb((db) =>
						resolveFocalEntities(db, agentId, { queryTokens }),
					);

					if (focal.entityIds.length > 0) {
						const traversal = getDbAccessor().withReadDb((db) =>
							traverseKnowledgeGraph(focal.entityIds, db, agentId, {
								maxAspectsPerEntity: traversalCfg.maxAspectsPerEntity,
								maxAttributesPerAspect: traversalCfg.maxAttributesPerAspect,
								maxDependencyHops: traversalCfg.maxDependencyHops,
								minDependencyStrength: traversalCfg.minDependencyStrength,
								maxBranching: traversalCfg.maxBranching,
								maxTraversalPaths: traversalCfg.maxTraversalPaths,
								minConfidence: traversalCfg.minConfidence,
								timeoutMs: traversalCfg.timeoutMs,
							}),
						);

						const tw = traversalCfg.boostWeight;
						const scoredById = new Map(scored.map((row) => [row.id, row]));
						const missingIds: string[] = [];

						for (const memoryId of traversal.memoryIds) {
							const existing = scoredById.get(memoryId);
							if (existing) {
								existing.score = (1 - tw) * existing.score + tw;
							} else {
								missingIds.push(memoryId);
							}
						}

						if (missingIds.length > 0) {
							const placeholders = missingIds.map(() => "?").join(", ");
							const baseRows = getDbAccessor().withReadDb(
								(db) =>
									db
										.prepare(
											`SELECT
												 m.id,
												 COALESCE(MAX(ea.importance), m.importance, 0.5) AS traversal_score
											 FROM memories m
											 LEFT JOIN entity_attributes ea
											   ON ea.memory_id = m.id
											  AND ea.agent_id = ?
											  AND ea.status = 'active'
											 WHERE m.id IN (${placeholders})
											   AND m.is_deleted = 0
											 ${filter.sql}
											 GROUP BY m.id, m.importance`,
										)
										.all(agentId, ...missingIds, ...filter.args) as Array<{
										id: string;
										traversal_score: number;
									}>,
							);

							for (const row of baseRows) {
								scored.push({
									id: row.id,
									score: Math.max(minScore, Math.min(1, row.traversal_score)),
									source: "ka_traversal",
								});
							}
						}

						scored.sort((a, b) => b.score - a.score);

						setTraversalStatus({
							phase: "recall",
							at: new Date().toISOString(),
							source: focal.source,
							focalEntityNames: focal.entityNames,
							focalEntities: focal.entityIds.length,
							traversedEntities: traversal.entityCount,
							memoryCount: traversal.memoryIds.size,
							constraintCount: traversal.constraints.length,
							timedOut: traversal.timedOut,
						});
					}
				}
			} catch (e) {
				logger.warn("memory", "KA traversal boost failed (non-fatal)", {
					error: e instanceof Error ? e.message : String(e),
				});
			}
		}
	}

	// --- Optional reranker hook ---
	if (cfg.pipelineV2.reranker.enabled) {
		try {
			const topForRerank = scored.slice(0, cfg.pipelineV2.reranker.topN);
			const rerankIds = topForRerank.map((s) => s.id);
			const rerankPlaceholders = rerankIds.map(() => "?").join(", ");

			// Fetch content for reranker — cross-encoders need document text
			const contentRows = getDbAccessor().withReadDb(
				(db) =>
					db
						.prepare(
							`SELECT id, content FROM memories
							 WHERE id IN (${rerankPlaceholders})`,
						)
						.all(...rerankIds) as Array<{
						id: string;
						content: string;
					}>,
			);
			const contentMap = new Map(contentRows.map((r) => [r.id, r.content]));

			const candidates: RerankCandidate[] = topForRerank.map((s) => ({
				id: s.id,
				content: contentMap.get(s.id) ?? "",
				score: s.score,
			}));
			// Use embedding reranker when query vector is available, else noop
			const provider = queryVecF32 ? createEmbeddingReranker(getDbAccessor(), queryVecF32) : noopReranker;
			const reranked = await rerank(query, candidates, provider, {
				topN: cfg.pipelineV2.reranker.topN,
				timeoutMs: cfg.pipelineV2.reranker.timeoutMs,
				model: cfg.pipelineV2.reranker.model,
			});
			// Update scores from reranked results
			const rerankedMap = new Map(reranked.map((r, i) => [r.id, i]));
			for (const s of scored) {
				const idx = rerankedMap.get(s.id);
				if (idx !== undefined) {
					// Preserve relative order from reranker
					s.score = 1 - idx / reranked.length;
				}
			}
			scored.sort((a, b) => b.score - a.score);
		} catch (e) {
			logger.warn("memory", "Reranker failed (non-fatal)", {
				error: e instanceof Error ? e.message : String(e),
			});
		}
	}

	// Over-fetch before hydration when scoped. With vector search
	// skipped, all candidates should already be in-scope, but this
	// guards against edge cases in traversal or graph boost.
	const preHydrate = scoped ? limit * 3 : limit;
	const topIds = scored.slice(0, preHydrate).map((s) => s.id);

	if (topIds.length === 0) {
		return { results: [], query, method: "hybrid" };
	}

	// --- Fetch full memory rows ---
	// Scope filter on hydration catches any results that bypassed
	// the FTS filter clause (e.g. unscoped graph boost results).
	const scopeClause =
		params.scope !== undefined
			? params.scope === null
				? " AND scope IS NULL"
				: " AND scope = ?"
			: " AND scope IS NULL";
	const scopeArgs: unknown[] =
		params.scope !== undefined && params.scope !== null ? [params.scope] : [];
	const placeholders = topIds.map(() => "?").join(", ");

	const rows = getDbAccessor().withReadDb(
		(db) =>
			db
				.prepare(
					`SELECT id, content, type, tags, pinned, importance, who, project, created_at
        FROM memories
        WHERE id IN (${placeholders})${scopeClause}`,
				)
				.all(...topIds, ...scopeArgs) as Array<{
				id: string;
				content: string;
				type: string;
				tags: string | null;
				pinned: number;
				importance: number;
				who: string;
				project: string | null;
				created_at: string;
			}>,
	);

	// Update access tracking (don't fail if this fails)
	try {
		getDbAccessor().withWriteTx((db) => {
			db.prepare(
				`UPDATE memories
          SET last_accessed = datetime('now'), access_count = access_count + 1
          WHERE id IN (${placeholders})`,
			).run(...topIds);
		});
	} catch (e) {
		logger.warn("memory", "Failed to update access tracking", e as Error);
	}

	const rowMap = new Map(rows.map((r) => [r.id, r]));
	const recallTruncate = cfg.pipelineV2.guardrails.recallTruncateChars;
	const results: RecallResult[] = scored
		.slice(0, limit)
		.filter((s) => rowMap.has(s.id))
		.map((s) => {
			const r = rowMap.get(s.id)!;
			const isTruncated = r.content.length > recallTruncate;
			return {
				id: r.id,
				content: isTruncated ? `${r.content.slice(0, recallTruncate)} [truncated]` : r.content,
				content_length: r.content.length,
				truncated: isTruncated,
				score: Math.round(s.score * 100) / 100,
				source: s.source,
				type: r.type,
				tags: r.tags,
				pinned: !!r.pinned,
				importance: r.importance,
				who: r.who,
				project: r.project,
				created_at: r.created_at,
			};
		});

	// --- Decision-rationale linking: auto-fetch linked rationale memories ---
	const decisionIds = results.filter((r) => r.type === "decision").map((r) => r.id);
	const existingIds = new Set(results.map((r) => r.id));

	if (decisionIds.length > 0 && cfg.pipelineV2.graph.enabled) {
		try {
			const supplementary = getDbAccessor().withReadDb((db) => {
				// Find entities linked to decision memories
				const dPlaceholders = decisionIds.map(() => "?").join(", ");
				const entityIds = db
					.prepare(
						`SELECT DISTINCT entity_id FROM memory_entity_mentions
							 WHERE memory_id IN (${dPlaceholders})`,
					)
					.all(...decisionIds) as Array<{ entity_id: string }>;

				if (entityIds.length === 0) return [];

				// Find rationale memories linked to same entities
				const ePlaceholders = entityIds.map(() => "?").join(", ");
				const eIds = entityIds.map((r) => r.entity_id);

				return db
					.prepare(
						`SELECT DISTINCT m.id, m.content, m.type, m.tags, m.pinned,
							        m.importance, m.who, m.project, m.created_at
							 FROM memory_entity_mentions mem
							 JOIN memories m ON m.id = mem.memory_id
							 WHERE mem.entity_id IN (${ePlaceholders})
							   AND m.type = 'rationale'
							   AND m.is_deleted = 0
							   ${scopeClause}
							 LIMIT 10`,
					)
					.all(...eIds, ...scopeArgs) as Array<{
					id: string;
					content: string;
					type: string;
					tags: string | null;
					pinned: number;
					importance: number;
					who: string;
					project: string | null;
					created_at: string;
				}>;
			});

			for (const r of supplementary) {
				if (existingIds.has(r.id)) continue;
				existingIds.add(r.id);
				const isTrunc = r.content.length > recallTruncate;
				results.push({
					id: r.id,
					content: isTrunc ? `${r.content.slice(0, recallTruncate)} [truncated]` : r.content,
					content_length: r.content.length,
					truncated: isTrunc,
					score: 0,
					source: "graph",
					type: r.type,
					tags: r.tags,
					pinned: !!r.pinned,
					importance: r.importance,
					who: r.who,
					project: r.project,
					created_at: r.created_at,
					supplementary: true,
				});
			}
		} catch (e) {
			logger.warn("memory", "Rationale linking failed (non-fatal)", {
				error: e instanceof Error ? e.message : String(e),
			});
		}
	}

	// --- Entity context + constructed memories (DP-7) ---
	let entityContext: RecallResponse["entities"];
	let focalEids: string[] = [];

	if (cfg.pipelineV2.graph.enabled && cfg.pipelineV2.traversal?.enabled) {
		try {
			const queryTokens = tokenizeGraphQuery(query);
			if (queryTokens.length > 0) {
				const agentId = params.agentId ?? "default";
				const ctx = getDbAccessor().withReadDb((db) => {
					const focal = resolveFocalEntities(db, agentId, { queryTokens });
					if (focal.entityIds.length === 0) return null;

					// Scope-filter: only include entities mentioned in
					// in-scope memories so unscoped entities (codebase
					// concepts etc.) don't pollute scoped searches.
					let eids = focal.entityIds;
					if (params.scope !== undefined) {
						const ph = eids.map(() => "?").join(", ");
						const sc = params.scope === null ? "m.scope IS NULL" : "m.scope = ?";
						const sa: unknown[] = params.scope === null ? [] : [params.scope];
						const sr = db
							.prepare(
								`SELECT DISTINCT mem.entity_id
								 FROM memory_entity_mentions mem
								 JOIN memories m ON m.id = mem.memory_id
								 WHERE mem.entity_id IN (${ph})
								   AND ${sc} AND m.is_deleted = 0`,
							)
							.all(...eids, ...sa) as Array<{ entity_id: string }>;
						eids = sr.map((r) => r.entity_id);
						if (eids.length === 0) return null;
					}

					const placeholders = eids.map(() => "?").join(", ");
					const entities = db
						.prepare(
							`SELECT id, name, entity_type FROM entities
							 WHERE id IN (${placeholders})`,
						)
						.all(...eids) as Array<{
						id: string;
						name: string;
						entity_type: string;
					}>;

					const structured = entities.map((ent) => {
						const aspects = db
							.prepare(
								`SELECT id, name FROM entity_aspects
								 WHERE entity_id = ? AND agent_id = ?
								 ORDER BY weight DESC LIMIT 10`,
							)
							.all(ent.id, agentId) as Array<{ id: string; name: string }>;

						return {
							name: ent.name,
							type: ent.entity_type,
							aspects: aspects.map((asp) => {
								const attrs = db
									.prepare(
										`SELECT content, status, importance FROM entity_attributes
										 WHERE aspect_id = ? AND agent_id = ? AND status = 'active'
										 ORDER BY importance DESC LIMIT 5`,
									)
									.all(asp.id, agentId) as Array<{
									content: string;
									status: string;
									importance: number;
								}>;
								return { name: asp.name, attributes: attrs };
							}).filter((a) => a.attributes.length > 0),
						};
					}).filter((e) => e.aspects.length > 0);

					return { eids, structured };
				});

				if (ctx) {
					entityContext = ctx.structured;
					focalEids = ctx.eids;
				}
			}
		} catch (e) {
			logger.warn("memory", "Entity context fetch failed (non-fatal)", {
				error: e instanceof Error ? e.message : String(e),
			});
		}
	}

	// --- Constructed memories: synthesize from graph paths (DP-7) ---
	if (focalEids.length > 0) {
		try {
			const agentId = params.agentId ?? "default";
			const cap = Math.max(3, Math.ceil(limit * 0.3));
			const blocks = getDbAccessor().withReadDb((db) =>
				constructContextBlocks(db, agentId, focalEids, cap),
			);
			const now = new Date().toISOString();
			let added = 0;
			for (const block of blocks) {
				if (added >= cap) break;
				const syntheticId = `constructed:${block.provenance.entityName}`;
				if (existingIds.has(syntheticId)) continue;
				existingIds.add(syntheticId);
				added++;

				results.push({
					id: syntheticId,
					content: block.content,
					content_length: block.content.length,
					truncated: false,
					score: Math.round(block.score * 100) / 100,
					source: "constructed",
					type: "semantic",
					tags: null,
					pinned: false,
					importance: 0.85,
					who: "",
					project: null,
					created_at: now,
					supplementary: true,
				});
			}
		} catch (e) {
			logger.warn("memory", "Constructed context failed (non-fatal)", {
				error: e instanceof Error ? e.message : String(e),
			});
		}
	}

	return {
		results,
		query,
		method: vectorMap.size > 0 ? "hybrid" : "keyword",
		entities: entityContext && entityContext.length > 0 ? entityContext : undefined,
	};
}
