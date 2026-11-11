import { getDbAccessorPath } from "./db-accessor";
import { getDbOwner } from "./db-owner-runtime";
import { ownerReadAll } from "./db-owner-sql";

/**
 * Cross-source identity lookup.
 *
 * Read-time, derived view over source artifacts: nothing is stored and no
 * semantic claim is written. Two records are reported as the same person only
 * when they share a verified email address or phone number, or when one names
 * the other's external ID. A name-only match is reported separately as
 * `possible` so a person can confirm it; it is never treated as confirmed.
 */

export type IdentityKeyType = "email" | "phone";

export interface IdentityKey {
	readonly type: IdentityKeyType;
	readonly value: string;
}

export type IdentityMatch = "query" | "email" | "phone" | "id";

export interface IdentityRecord {
	readonly sourcePath: string;
	readonly sourceKind: string;
	readonly sourceId: string | null;
	readonly provider: string;
	readonly title: string;
	readonly match: IdentityMatch;
	readonly matchedValue: string;
}

export interface PossibleIdentityRecord {
	readonly sourcePath: string;
	readonly sourceKind: string;
	readonly sourceId: string | null;
	readonly provider: string;
	readonly title: string;
	readonly reason: string;
}

export interface IdentityLookupResult {
	readonly query: string;
	readonly keys: readonly IdentityKey[];
	readonly records: readonly IdentityRecord[];
	readonly possible: readonly PossibleIdentityRecord[];
	/** True when a bound was reached and more matching records may exist. */
	readonly truncated: boolean;
}

export interface IdentityLookupParams {
	readonly agentId: string;
	readonly query: string;
}

export const IDENTITY_PERSON_KINDS: readonly string[] = ["source_stripe_customer", "source_gohighlevel_contact"];
export const IDENTITY_MAX_QUERY_CHARS = 200;
const MAX_SEEDS = 5;
const MAX_KEYS = 12;
const MAX_IDS = 12;
const ROWS_PER_SEARCH = 50;
const MAX_RECORDS = 200;
const MAX_POSSIBLE = 25;
const EXPANSION_ROUNDS = 3;

const EMAIL_PATTERN = /[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+/gi;
const FIELD_LINE = /^- ([^:\n]{1,60}): (.+)$/gm;

interface ArtifactRow {
	readonly source_path: string;
	readonly source_kind: string;
	readonly source_id: string | null;
	readonly content: string;
}

export function normalizeEmail(value: string): string | null {
	const match = value.trim().toLowerCase().match(EMAIL_PATTERN);
	return match?.length === 1 && match[0] === value.trim().toLowerCase() ? match[0] : null;
}

/** Phone numbers compare on their last ten digits; shorter numbers are not used as keys. */
export function normalizePhone(value: string): string | null {
	if (!/^[\d\s()+.-]{7,24}$/.test(value.trim())) return null;
	const digits = value.replace(/\D/g, "");
	return digits.length >= 10 && digits.length <= 15 ? digits.slice(-10) : null;
}

export function extractIdentityKeys(content: string): IdentityKey[] {
	const keys = new Map<string, IdentityKey>();
	for (const line of content.matchAll(FIELD_LINE)) {
		const label = (line[1] ?? "").toLowerCase();
		const value = line[2] ?? "";
		if (label.includes("email")) {
			for (const found of value.toLowerCase().match(EMAIL_PATTERN) ?? [])
				keys.set(`email:${found}`, { type: "email", value: found });
		} else if (label.includes("phone")) {
			for (const part of value.split(/[,;]/)) {
				const phone = normalizePhone(part);
				if (phone) keys.set(`phone:${phone}`, { type: "phone", value: phone });
			}
		}
	}
	return [...keys.values()];
}

function titleOf(content: string): string {
	return /^#\s+(.+?)\s*$/m.exec(content)?.[1]?.trim() ?? "";
}

function normalizeName(value: string): string {
	return value
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, " ")
		.trim();
}

function providerOf(sourceKind: string): string {
	return sourceKind.replace(/^source_/, "").split("_")[0] ?? sourceKind;
}

function externalIdOf(sourcePath: string): string | null {
	const tail = sourcePath.split("/").at(-1) ?? "";
	return /^[A-Za-z0-9_-]{8,64}$/.test(tail) ? tail : null;
}

function tokens(value: string): string[] {
	return value.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

function phrase(value: string): string | null {
	const parts = tokens(value);
	return parts.length > 0 ? `"${parts.join(" ")}"` : null;
}

function ftsForKey(key: IdentityKey): string | null {
	if (key.type === "email") return phrase(key.value);
	const d = key.value;
	return [
		`"${d}"`,
		`"1${d}"`,
		`"${d.slice(0, 3)} ${d.slice(3, 6)} ${d.slice(6)}"`,
		`"${d.slice(0, 3)} ${d.slice(3)}"`,
	].join(" OR ");
}

async function search(agentId: string, fts: string, personOnly: boolean): Promise<readonly ArtifactRow[]> {
	const kinds = personOnly
		? `AND ma.source_kind IN (${IDENTITY_PERSON_KINDS.map(() => "?").join(", ")})`
		: "AND ma.source_kind LIKE 'source_%' AND ma.source_kind NOT LIKE '%_failure'";
	return await ownerReadAll<ArtifactRow>(
		await getDbOwner(getDbAccessorPath()),
		`SELECT ma.source_path, ma.source_kind, ma.source_id, ma.content
		 FROM memory_artifacts_fts
		 JOIN memory_artifacts ma ON ma.rowid = memory_artifacts_fts.rowid
		 WHERE memory_artifacts_fts MATCH ?
		   AND ma.agent_id = ?
		   AND COALESCE(ma.is_deleted, 0) = 0
		   ${kinds}
		 ORDER BY bm25(memory_artifacts_fts) ASC, ma.source_path ASC LIMIT ?`,
		[fts, agentId, ...(personOnly ? IDENTITY_PERSON_KINDS : []), ROWS_PER_SEARCH + 1],
		{
			operation: "sources.identity.lookup",
			workloadClass: "foreground",
			estimatedWorkUnits: ROWS_PER_SEARCH,
			deadlineMs: 30_000,
		},
	);
}

function keyOfQuery(query: string): IdentityKey | null {
	const email = normalizeEmail(query);
	if (email) return { type: "email", value: email };
	const phone = normalizePhone(query);
	return phone ? { type: "phone", value: phone } : null;
}

export async function lookupIdentity(params: IdentityLookupParams): Promise<IdentityLookupResult> {
	const query = params.query.trim().slice(0, IDENTITY_MAX_QUERY_CHARS);
	const keys = new Map<string, IdentityKey>();
	const ids = new Map<string, string>();
	const records = new Map<string, IdentityRecord>();
	const names = new Set<string>();
	let truncated = false;

	const accept = (row: ArtifactRow, match: IdentityMatch, matchedValue: string): boolean => {
		if (records.has(row.source_path)) return false;
		if (records.size >= MAX_RECORDS) {
			truncated = true;
			return false;
		}
		const title = titleOf(row.content);
		records.set(row.source_path, {
			sourcePath: row.source_path,
			sourceKind: row.source_kind,
			sourceId: row.source_id,
			provider: providerOf(row.source_kind),
			title,
			match,
			matchedValue,
		});
		if (!IDENTITY_PERSON_KINDS.includes(row.source_kind)) return true;
		const name = normalizeName(title);
		if (name.includes(" ")) names.add(name);
		const externalId = externalIdOf(row.source_path);
		if (externalId && !ids.has(externalId)) {
			if (ids.size < MAX_IDS) ids.set(externalId, externalId);
			else truncated = true;
		}
		for (const key of extractIdentityKeys(row.content)) {
			const id = `${key.type}:${key.value}`;
			if (keys.has(id)) continue;
			if (keys.size < MAX_KEYS) keys.set(id, key);
			else truncated = true;
		}
		return true;
	};

	const direct = keyOfQuery(query);
	if (direct) keys.set(`${direct.type}:${direct.value}`, direct);
	else {
		const wanted = tokens(query);
		const fts = wanted.map((token) => `"${token}"`).join(" ");
		if (fts.length === 0) return { query, keys: [], records: [], possible: [], truncated: false };
		const rows = await search(params.agentId, fts, true);
		const exactId = rows.filter((row) => externalIdOf(row.source_path)?.toLowerCase() === query.toLowerCase());
		const byName = rows.filter((row) => normalizeName(titleOf(row.content)) === normalizeName(query));
		const seeds = exactId.length > 0 ? exactId : byName;
		if (seeds.length > MAX_SEEDS) truncated = true;
		for (const row of seeds.slice(0, MAX_SEEDS)) accept(row, "query", query);
	}

	const searchedKeys = new Set<string>();
	for (let round = 0; round < EXPANSION_ROUNDS; round++) {
		const pending = [...keys.entries()].filter(([id]) => !searchedKeys.has(id));
		if (pending.length === 0) break;
		for (const [id, key] of pending) {
			searchedKeys.add(id);
			const fts = ftsForKey(key);
			if (!fts) continue;
			const rows = await search(params.agentId, fts, false);
			if (rows.length > ROWS_PER_SEARCH) truncated = true;
			for (const row of rows.slice(0, ROWS_PER_SEARCH)) {
				// FTS finds candidates; only an exact normalized key in a labelled field confirms.
				if (extractIdentityKeys(row.content).some((found) => found.type === key.type && found.value === key.value))
					accept(row, key.type, key.value);
			}
		}
	}

	if ([...keys.keys()].some((id) => !searchedKeys.has(id))) truncated = true;

	for (const externalId of ids.keys()) {
		const fts = phrase(externalId);
		if (!fts) continue;
		const rows = await search(params.agentId, fts, false);
		if (rows.length > ROWS_PER_SEARCH) truncated = true;
		const bounded = new RegExp(`(^|[^A-Za-z0-9_-])${externalId}([^A-Za-z0-9_-]|$)`);
		for (const row of rows.slice(0, ROWS_PER_SEARCH)) {
			if (IDENTITY_PERSON_KINDS.includes(row.source_kind)) continue;
			if (bounded.test(row.content)) accept(row, "id", externalId);
		}
	}

	const possible = new Map<string, PossibleIdentityRecord>();
	for (const name of names) {
		const fts = name
			.split(" ")
			.map((token) => `"${token}"`)
			.join(" ");
		const rows = await search(params.agentId, fts, true);
		for (const row of rows) {
			if (records.has(row.source_path) || possible.has(row.source_path)) continue;
			const title = titleOf(row.content);
			if (normalizeName(title) !== name) continue;
			if (possible.size >= MAX_POSSIBLE) {
				truncated = true;
				break;
			}
			possible.set(row.source_path, {
				sourcePath: row.source_path,
				sourceKind: row.source_kind,
				sourceId: row.source_id,
				provider: providerOf(row.source_kind),
				title,
				reason: "Same name, but no shared email address or phone number",
			});
		}
	}

	return {
		query,
		keys: [...keys.values()],
		records: [...records.values()],
		possible: [...possible.values()],
		truncated,
	};
}
