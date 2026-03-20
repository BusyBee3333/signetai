/**
 * DP-7: Constructed memories with path provenance.
 *
 * Synthesizes purpose-built context blocks from knowledge graph
 * traversal paths. Each block combines entity attributes, constraints,
 * and dependency relationships into a coherent text representation
 * with provenance metadata for future path feedback (DP-9).
 *
 * No LLM calls — pure template synthesis.
 */

import type { ReadDb } from "../db-accessor";

// ---------------------------------------------------------------------------
// Public interfaces
// ---------------------------------------------------------------------------

export interface ConstructedProvenance {
	readonly entityId: string;
	readonly entityName: string;
	readonly entityType: string;
	readonly aspectIds: ReadonlyArray<string>;
	readonly aspectNames: ReadonlyArray<string>;
	readonly attributeCount: number;
	readonly constraintCount: number;
	readonly dependencyEntityIds: ReadonlyArray<string>;
}

export interface ConstructedContext {
	readonly content: string;
	readonly score: number;
	readonly source: "constructed";
	readonly provenance: ConstructedProvenance;
}

// ---------------------------------------------------------------------------
// Internal row types
// ---------------------------------------------------------------------------

interface EntityRow {
	readonly id: string;
	readonly name: string;
	readonly entity_type: string;
}

interface AspectRow {
	readonly id: string;
	readonly name: string;
}

interface AttributeRow {
	readonly content: string;
	readonly importance: number;
}

interface ConstraintRow {
	readonly content: string;
	readonly importance: number;
}

interface DependencyRow {
	readonly target_entity_id: string;
	readonly name: string;
}

// ---------------------------------------------------------------------------
// Score normalization
// ---------------------------------------------------------------------------

/** Structural density score: more structure = higher score, clamped to [0, 1]. */
function densityScore(aspects: number, attrs: number, constraints: number): number {
	// Weighted sum: aspects contribute breadth, attributes depth,
	// constraints are high-value invariants worth extra weight.
	const raw = aspects * 0.15 + attrs * 0.05 + constraints * 0.2;
	return Math.min(1, Math.max(0, raw));
}

// ---------------------------------------------------------------------------
// Main construction function
// ---------------------------------------------------------------------------

export function constructContextBlocks(
	db: ReadDb,
	agentId: string,
	focalEntityIds: ReadonlyArray<string>,
	limit: number,
): ReadonlyArray<ConstructedContext> {
	if (focalEntityIds.length === 0) return [];

	const ph = focalEntityIds.map(() => "?").join(", ");
	const entities = db
		.prepare(
			`SELECT id, name, entity_type FROM entities
			 WHERE id IN (${ph})`,
		)
		.all(...focalEntityIds) as EntityRow[];

	if (entities.length === 0) return [];

	const blocks: ConstructedContext[] = [];

	for (const ent of entities) {
		const aspects = db
			.prepare(
				`SELECT id, name FROM entity_aspects
				 WHERE entity_id = ? AND agent_id = ?
				 ORDER BY weight DESC LIMIT 10`,
			)
			.all(ent.id, agentId) as AspectRow[];

		const lines: string[] = [];
		const aspectIds: string[] = [];
		const aspectNames: string[] = [];
		let totalAttrs = 0;

		for (const asp of aspects) {
			const attrs = db
				.prepare(
					`SELECT content, importance FROM entity_attributes
					 WHERE aspect_id = ? AND agent_id = ?
					   AND status = 'active' AND kind != 'constraint'
					 ORDER BY importance DESC LIMIT 5`,
				)
				.all(asp.id, agentId) as AttributeRow[];

			if (attrs.length === 0) continue;

			aspectIds.push(asp.id);
			aspectNames.push(asp.name);
			totalAttrs += attrs.length;

			const vals = attrs.map((a) => a.content).join("; ");
			lines.push(`- ${asp.name}: ${vals}`);
		}

		// Constraints: always surface (invariant 5)
		const constraints = db
			.prepare(
				`SELECT DISTINCT ea.content, ea.importance
				 FROM entity_attributes ea
				 JOIN entity_aspects asp ON asp.id = ea.aspect_id
				 WHERE asp.entity_id = ? AND ea.agent_id = ?
				   AND ea.kind = 'constraint' AND ea.status = 'active'
				 ORDER BY ea.importance DESC LIMIT 10`,
			)
			.all(ent.id, agentId) as ConstraintRow[];

		if (constraints.length > 0) {
			const vals = constraints.map((c) => c.content).join("; ");
			lines.push(`- Constraints: ${vals}`);
		}

		// Dependencies: cross-reference names
		const deps = db
			.prepare(
				`SELECT ed.target_entity_id, e.name
				 FROM entity_dependencies ed
				 JOIN entities e ON e.id = ed.target_entity_id
				 WHERE ed.source_entity_id = ? AND ed.agent_id = ?
				   AND ed.strength >= 0.3
				 ORDER BY ed.strength DESC LIMIT 8`,
			)
			.all(ent.id, agentId) as DependencyRow[];

		if (deps.length > 0) {
			lines.push(`- Related: ${deps.map((d) => d.name).join(", ")}`);
		}

		if (lines.length === 0) continue;

		const text = `[${ent.name} (${ent.entity_type})]\n${lines.join("\n")}`;
		const score = densityScore(aspectIds.length, totalAttrs, constraints.length);

		blocks.push({
			content: text,
			score,
			source: "constructed",
			provenance: {
				entityId: ent.id,
				entityName: ent.name,
				entityType: ent.entity_type,
				aspectIds,
				aspectNames,
				attributeCount: totalAttrs,
				constraintCount: constraints.length,
				dependencyEntityIds: deps.map((d) => d.target_entity_id),
			},
		});
	}

	// Sort by density score descending, then truncate to limit
	blocks.sort((a, b) => b.score - a.score);
	return blocks.slice(0, limit);
}
