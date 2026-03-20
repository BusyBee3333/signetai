import type { MigrationDb } from "./index";

/**
 * Rebuild memories_fts with porter stemming.
 *
 * The default unicode61 tokenizer does exact token matching, so
 * "interested" doesn't match "interest" and "camping" doesn't
 * match "camp". Porter stemming normalizes morphological variants,
 * significantly improving BM25 recall for natural language queries.
 *
 * This drops and recreates the FTS table + sync triggers.
 */
export function up(db: MigrationDb): void {
	// Drop old triggers first
	db.exec("DROP TRIGGER IF EXISTS memories_fts_ai");
	db.exec("DROP TRIGGER IF EXISTS memories_fts_ad");
	db.exec("DROP TRIGGER IF EXISTS memories_fts_au");

	// Drop and recreate with porter stemming
	db.exec("DROP TABLE IF EXISTS memories_fts");
	db.exec(`
		CREATE VIRTUAL TABLE memories_fts USING fts5(
			content,
			content='memories', content_rowid='rowid',
			tokenize='porter unicode61'
		)
	`);

	// Rebuild index from existing data
	db.exec(`
		INSERT INTO memories_fts(rowid, content)
		SELECT rowid, content FROM memories
	`);

	// Recreate sync triggers
	db.exec(`
		CREATE TRIGGER memories_fts_ai AFTER INSERT ON memories BEGIN
			INSERT INTO memories_fts(rowid, content)
			VALUES (new.rowid, new.content);
		END
	`);

	db.exec(`
		CREATE TRIGGER memories_fts_ad AFTER DELETE ON memories BEGIN
			INSERT INTO memories_fts(memories_fts, rowid, content)
			VALUES ('delete', old.rowid, old.content);
		END
	`);

	db.exec(`
		CREATE TRIGGER memories_fts_au AFTER UPDATE ON memories BEGIN
			INSERT INTO memories_fts(memories_fts, rowid, content)
			VALUES ('delete', old.rowid, old.content);
			INSERT INTO memories_fts(rowid, content)
			VALUES (new.rowid, new.content);
		END
	`);
}
