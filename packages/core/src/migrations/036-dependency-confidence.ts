import type { MigrationDb } from "./index";

/**
 * Migration 036: Add confidence column to entity_dependencies.
 *
 * Combined (COALESCE(confidence, 0.7) * strength) score gates traversal
 * so the walk prefers trustworthy edges. Reason values and their default
 * confidence:
 *   user-asserted (1.0), multi-memory (0.9), single-memory (0.7),
 *   pattern-matched (0.5), inferred (0.4), llm-uncertain (0.3).
 *
 * The reason column was added in migration 031; this migration adds only
 * the confidence column with DEFAULT 0.7 (single-memory).
 */
export function up(db: MigrationDb): void {
	const cols = db.prepare("PRAGMA table_info(entity_dependencies)").all() as Array<{ name: string }>;

	if (!cols.some((c) => c.name === "confidence")) {
		db.exec("ALTER TABLE entity_dependencies ADD COLUMN confidence REAL DEFAULT 0.7");
	}
}
