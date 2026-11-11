import { createHash } from "node:crypto";
import {
	type SignetSourceEntry,
	type SourceFailureState,
	type SourceProviderKind,
	type GoHighLevelSourceResourceType,
	parseGoHighLevelSettings,
} from "@signet/core";
import { yieldEvery } from "./async-yield";
import { dbOwnerQuery, dbOwnerTransaction, ownerStatement } from "./db-owner-runtime";
import { logger } from "./logger";
import { indexExternalMemoryArtifact } from "./memory-lineage";
import { getSecret } from "./secrets";
import { indexSourceArtifactStructureAsync, purgeSourceArtifactStructureAsync } from "./source-artifact-graph";
import type { SourceProviderAdapter, SourceProviderSyncContext, SourceProviderSyncResult } from "./source-providers";
import { purgeSourceOwnedRows } from "./source-purge";
import {
	GOHIGHLEVEL_PAGE_SIZE,
	type GoHighLevelCursor,
	type GoHighLevelListPage,
	type GoHighLevelRecord,
	GoHighLevelRequestError,
	type GoHighLevelTranscript,
	fetchGoHighLevelTranscript,
	listGoHighLevelPage,
} from "./gohighlevel-source-fetch";

const GOHIGHLEVEL_PROVIDER_KIND: SourceProviderKind = "gohighlevel";
const GOHIGHLEVEL_HARNESS = "gohighlevel";
const FAILURE_KIND = "source_gohighlevel_failure";
export const GOHIGHLEVEL_MAX_CONSECUTIVE_REQUEST_FAILURES = 5;
export const GOHIGHLEVEL_SYNC_DEADLINE_MS = 3 * 60 * 60 * 1000;
const STALE_SCAN_BATCH = 500;
const PURGE_BATCH = 100;

const ARTIFACT_KINDS: Readonly<Record<GoHighLevelSourceResourceType, string>> = {
	contacts: "source_gohighlevel_contact",
	opportunities: "source_gohighlevel_opportunity",
	conversations: "source_gohighlevel_conversation",
};

let syncDeadlineMs = GOHIGHLEVEL_SYNC_DEADLINE_MS;

export function setGoHighLevelSyncDeadlineForTest(ms: number | null): void {
	syncDeadlineMs = ms ?? GOHIGHLEVEL_SYNC_DEADLINE_MS;
}

type StopReason = { readonly message: string; readonly haltsSync: boolean };

interface IndexedVersion {
	readonly contentHash: string | null;
	readonly versionKey: string | null;
}

interface ResourceOutcome {
	readonly scanned: number;
	readonly indexed: number;
	readonly complete: boolean;
	readonly seenPaths: ReadonlySet<string>;
	readonly haltSync: string | null;
}

export const goHighLevelSourceProvider: SourceProviderAdapter = {
	kind: GOHIGHLEVEL_PROVIDER_KIND,
	sync: syncGoHighLevelSource,
	purge: async (source, agentId) => await purgeSourceOwnedRows({ sourceId: source.id, agentId }),
};

async function syncGoHighLevelSource(context: SourceProviderSyncContext): Promise<SourceProviderSyncResult> {
	const source = context.source;
	const agentId = context.agentId;
	const settings = parseGoHighLevelSettings(source.providerSettings);
	const syncStartedAt = new Date().toISOString();
	const token = await resolveToken(settings.tokenRef);
	const failures: SourceFailureState[] = [];
	const deadline = Date.now() + syncDeadlineMs;
	let indexed = 0;
	let scanned = 0;
	context.onProgress?.({ scanned, total: scanned, indexed, currentPath: source.root });

	for (const resource of settings.resourceTypes) {
		if (!context.shouldContinue()) break;
		const outcome = await syncResource({
			context,
			token,
			locationId: settings.locationId,
			resource,
			maxItems: settings.maxItemsPerType,
			deadline,
			failures,
			scannedBefore: scanned,
			indexedBefore: indexed,
		});
		scanned += outcome.scanned;
		indexed += outcome.indexed;
		if (outcome.complete && context.shouldContinue()) {
			try {
				await purgeStaleRecords(source, agentId, resource, outcome.seenPaths, syncStartedAt, context.shouldContinue);
			} catch (err) {
				failures.push(
					failureState(source, `GoHighLevel stale ${resource} purge failed: ${errorMessage(err)}`, {
						phase: "purge",
						resource,
					}),
				);
			}
		}
		if (outcome.haltSync) {
			const remaining = settings.resourceTypes.slice(settings.resourceTypes.indexOf(resource) + 1);
			if (remaining.length > 0 && context.shouldContinue()) {
				failures.push(
					failureState(source, `${outcome.haltSync}; ${remaining.join(", ")} were not attempted`, {
						phase: "sync",
						notAttempted: remaining,
					}),
				);
			}
			break;
		}
	}

	context.onProgress?.({ scanned, total: scanned, indexed, currentPath: source.root });
	if (!context.shouldContinue()) return { indexed, scanned, total: scanned, failures };
	await purgeStaleFailureArtifacts(source.id, agentId, syncStartedAt);
	for (const failure of failures) {
		await writeFailureArtifact(source, agentId, failure);
	}
	return { indexed: indexed + failures.length, scanned, total: scanned, failures };
}

async function syncResource(input: {
	readonly context: SourceProviderSyncContext;
	readonly token: string;
	readonly locationId: string;
	readonly resource: GoHighLevelSourceResourceType;
	readonly maxItems: number;
	readonly deadline: number;
	readonly failures: SourceFailureState[];
	readonly scannedBefore: number;
	readonly indexedBefore: number;
}): Promise<ResourceOutcome> {
	const { context, token, locationId, resource, maxItems, deadline, failures } = input;
	const source = context.source;
	const agentId = context.agentId;
	const seenPaths = new Set<string>();
	const yielder = yieldEvery(5);
	let scanned = 0;
	let indexed = 0;
	let unparsed = 0;
	let cursor: GoHighLevelCursor | null = null;
	let enumerated = 0;
	let expectedTotal: number | null = null;
	let reachedEnd = false;
	let stop: StopReason | null = null;
	let consecutiveRequestFailures = 0;
	let unfetched = 0;

	while (!reachedEnd && stop === null && scanned < maxItems && context.shouldContinue()) {
		if (Date.now() > deadline) {
			stop = {
				message: `GoHighLevel sync reached its ${Math.round(syncDeadlineMs / 60_000)}-minute deadline`,
				haltsSync: true,
			};
			break;
		}
		let page: GoHighLevelListPage;
		try {
			page = await listGoHighLevelPage(
				token,
				locationId,
				resource,
				cursor,
				Math.min(GOHIGHLEVEL_PAGE_SIZE, maxItems - scanned),
				() => context.shouldContinue(),
			);
			consecutiveRequestFailures = 0;
		} catch (err) {
			if (!context.shouldContinue()) break;
			const kind = requestOutcome(err);
			if (kind === "unauthorized") {
				stop = { message: "GoHighLevel rejected the token", haltsSync: true };
			} else if (kind === "forbidden") {
				stop = { message: `The GoHighLevel token cannot read ${resource}: ${errorMessage(err)}`, haltsSync: false };
			} else if (kind === "transient") {
				consecutiveRequestFailures++;
				if (consecutiveRequestFailures < GOHIGHLEVEL_MAX_CONSECUTIVE_REQUEST_FAILURES) continue;
				stop = {
					message: `Stopped after ${consecutiveRequestFailures} consecutive GoHighLevel request failures: ${errorMessage(err)}`,
					haltsSync: true,
				};
			} else {
				stop = { message: `GoHighLevel ${resource} list failed: ${errorMessage(err)}`, haltsSync: false };
			}
			break;
		}
		unparsed += page.unparsed;
		enumerated += page.records.length + page.unparsed;
		if (expectedTotal === null) expectedTotal = page.total;
		let versions: ReadonlyMap<string, IndexedVersion>;
		try {
			versions = await readIndexedVersions(
				source.id,
				agentId,
				ARTIFACT_KINDS[resource],
				page.records.map((record) => recordPath(source, record)),
			);
		} catch (err) {
			stop = { message: `GoHighLevel index state read failed: ${errorMessage(err)}`, haltsSync: true };
			break;
		}
		for (const record of page.records) {
			if (!context.shouldContinue()) break;
			const path = recordPath(source, record);
			seenPaths.add(path);
			context.onProgress?.({
				scanned: input.scannedBefore + scanned,
				total: input.scannedBefore + scanned,
				indexed: input.indexedBefore + indexed,
				currentPath: path,
			});
			const stored = versions.get(path);
			let transcript: GoHighLevelTranscript | null = null;
			if (resource === "conversations") {
				if (stored && record.versionKey !== null && stored.versionKey === record.versionKey) {
					indexed++;
					scanned++;
					await yielder();
					continue;
				}
				try {
					transcript = await fetchGoHighLevelTranscript(token, record.id, () => context.shouldContinue());
					consecutiveRequestFailures = 0;
				} catch (err) {
					if (!context.shouldContinue()) break;
					const kind = requestOutcome(err);
					if (kind === "unauthorized") {
						stop = { message: "GoHighLevel rejected the token", haltsSync: true };
						break;
					}
					if (kind === "forbidden") {
						stop = {
							message: `The GoHighLevel token cannot read conversation messages: ${errorMessage(err)}`,
							haltsSync: false,
						};
						break;
					}
					unfetched++;
					if (kind === "transient") {
						consecutiveRequestFailures++;
						if (consecutiveRequestFailures >= GOHIGHLEVEL_MAX_CONSECUTIVE_REQUEST_FAILURES) {
							stop = {
								message: `Stopped after ${consecutiveRequestFailures} consecutive GoHighLevel request failures: ${errorMessage(err)}`,
								haltsSync: true,
							};
							break;
						}
					}
					scanned++;
					await yielder();
					continue;
				}
			}
			const content = recordContent(record, transcript);
			const hash = contentHash(content);
			if (stored?.contentHash === hash && stored.versionKey === record.versionKey) {
				indexed++;
			} else {
				try {
					await writeRecordArtifact(source, agentId, record, content, hash, transcript);
					indexed++;
				} catch (err) {
					failures.push(
						failureState(
							source,
							`GoHighLevel ${record.resource} indexing failed for ${record.id}: ${errorMessage(err)}`,
							{
								phase: "index",
								resource,
								objectId: record.id,
							},
						),
					);
				}
			}
			scanned++;
			await yielder();
		}
		if (stop !== null) break;
		if (page.next === null) reachedEnd = true;
		else cursor = page.next;
	}

	if (unfetched > 0 && context.shouldContinue()) {
		failures.push(
			failureState(
				source,
				`${unfetched} GoHighLevel conversation(s) could not be fetched and will be retried on the next sync`,
				{ phase: "messages", resource, unfetched },
			),
		);
	}
	if (unparsed > 0) {
		failures.push(
			failureState(source, `Skipped ${unparsed} GoHighLevel ${resource} object(s) with an unexpected shape`, {
				phase: "parse",
				resource,
				unparsed,
			}),
		);
	}
	if (reachedEnd && stop === null && expectedTotal !== null && enumerated < expectedTotal) {
		stop = {
			message: `GoHighLevel ${resource} enumeration ended after ${enumerated} of ${expectedTotal} reported objects`,
			haltsSync: false,
		};
	}
	const capped = !reachedEnd && stop === null && scanned >= maxItems;
	if (capped) {
		logger.warn("gohighlevel-source", "GoHighLevel enumeration hit configured cap", {
			sourceId: source.id,
			resource,
			maxItemsPerType: maxItems,
		});
	}
	if (stop && context.shouldContinue()) {
		failures.push(
			failureState(source, `${stop.message}; stale ${resource} were not purged`, {
				phase: "list",
				resource,
				scanned,
			}),
		);
	}
	return {
		scanned,
		indexed,
		complete: reachedEnd && stop === null && unparsed === 0,
		seenPaths,
		haltSync: stop?.haltsSync ? stop.message : null,
	};
}

function requestOutcome(err: unknown): "unauthorized" | "forbidden" | "transient" | "failed" {
	if (!(err instanceof GoHighLevelRequestError)) return "failed";
	if (err.status === 401) return "unauthorized";
	if (err.status === 403) return "forbidden";
	if (err.code !== "cancelled" && (err.retryable || err.status === 0)) return "transient";
	return "failed";
}

export function contentHash(content: string): string {
	return createHash("sha256").update(content).digest("hex").slice(0, 32);
}

async function writeRecordArtifact(
	source: SignetSourceEntry,
	agentId: string,
	record: GoHighLevelRecord,
	content: string,
	hash: string,
	transcript: GoHighLevelTranscript | null,
): Promise<void> {
	const sourcePath = recordPath(source, record);
	const sourceParentPath = collectionPath(source, record.resource);
	const sourceKind = ARTIFACT_KINDS[record.resource];
	const createdMs = record.updatedMs ?? record.createdMs ?? Date.now();
	await indexExternalMemoryArtifact({
		agentId,
		harness: GOHIGHLEVEL_HARNESS,
		sourceId: source.id,
		sourceRoot: source.root,
		sourceExternalId: record.id,
		sourceParentPath,
		sourcePath,
		sourceKind,
		sourceMtimeMs: createdMs,
		capturedAt: new Date(createdMs).toISOString(),
		content,
		sourceMeta: {
			provider: GOHIGHLEVEL_PROVIDER_KIND,
			resource: record.resource,
			objectId: record.id,
			contactId: record.contactId,
			contentHash: hash,
			versionKey: record.versionKey,
			...(transcript
				? {
						messages: transcript.messages,
						olderMessagesOmitted: transcript.olderMessagesOmitted,
						clipped: transcript.clipped,
						emailPreviewsOnly: transcript.emailPreviewsOnly,
					}
				: {}),
		},
	});
	await indexSourceArtifactStructureAsync({
		agentId,
		sourceId: source.id,
		sourceKind,
		sourceRoot: source.root,
		sourceParentPath,
		sourcePath,
		displayName: record.title,
		content,
	});
}

export function recordContent(record: GoHighLevelRecord, transcript: GoHighLevelTranscript | null = null): string {
	const lines = [`# ${record.title}`, "", `GoHighLevel ${singular(record.resource)}: ${record.id}`];
	if (record.fields.length > 0) {
		lines.push("");
		for (const entry of record.fields) lines.push(`- ${entry.name}: ${entry.value}`);
	}
	if (transcript) {
		lines.push("", "## Messages", "");
		lines.push(transcript.text || "No message text.");
		if (transcript.olderMessagesOmitted) {
			lines.push(
				"",
				`> Signet stored the ${transcript.messages} most recent messages; older messages are not included.`,
			);
		}
		if (transcript.clipped) lines.push("", "> Signet stored the beginning of this transcript only.");
		if (transcript.emailPreviewsOnly > 0) {
			lines.push("", `> ${transcript.emailPreviewsOnly} email(s) are shown as the preview GoHighLevel lists.`);
		}
	}
	return lines.join("\n");
}

function singular(resource: GoHighLevelSourceResourceType): string {
	return resource === "contacts" ? "contact" : resource === "opportunities" ? "opportunity" : "conversation";
}

function recordPath(source: SignetSourceEntry, record: GoHighLevelRecord): string {
	return `${collectionPath(source, record.resource)}/${record.id}`;
}

function collectionPath(source: SignetSourceEntry, resource: GoHighLevelSourceResourceType): string {
	return `gohighlevel://sources/${source.id}/${resource}`;
}

async function readIndexedVersions(
	sourceId: string,
	agentId: string,
	sourceKind: string,
	paths: readonly string[],
): Promise<ReadonlyMap<string, IndexedVersion>> {
	const hashes = new Map<string, IndexedVersion>();
	if (paths.length === 0) return hashes;
	const rows = await dbOwnerQuery<
		Array<{ readonly source_path: string; readonly content_hash: string | null; readonly version_key: string | null }>
	>(
		{
			sql: `SELECT source_path, json_extract(source_meta_json, '$.contentHash') AS content_hash,
			        json_extract(source_meta_json, '$.versionKey') AS version_key
			 FROM memory_artifacts
			 WHERE agent_id = ? AND source_id = ? AND source_kind = ?
			   AND COALESCE(is_deleted, 0) = 0
			   AND source_path IN (${paths.map(() => "?").join(", ")})`,
			params: [agentId, sourceId, sourceKind, ...paths],
			result: "all",
		},
		{ operation: "sources.gohighlevel.read_versions", lane: "read", deadlineMs: 5_000 },
	);
	for (const row of rows) {
		hashes.set(row.source_path, {
			contentHash: typeof row.content_hash === "string" ? row.content_hash : null,
			versionKey: typeof row.version_key === "string" ? row.version_key : null,
		});
	}
	return hashes;
}

async function purgeStaleRecords(
	source: SignetSourceEntry,
	agentId: string,
	resource: GoHighLevelSourceResourceType,
	seenPaths: ReadonlySet<string>,
	syncStartedAt: string,
	shouldContinue: () => boolean,
): Promise<void> {
	const stale: Array<{ readonly rowid: number; readonly sourcePath: string }> = [];
	let after = "";
	for (;;) {
		if (!shouldContinue()) return;
		const rows = await dbOwnerQuery<Array<{ readonly rowid: number; readonly source_path: string }>>(
			{
				sql: `SELECT rowid, source_path FROM memory_artifacts
				 WHERE agent_id = ? AND source_id = ? AND source_kind = ?
				   AND COALESCE(is_deleted, 0) = 0 AND source_path > ?
				 ORDER BY source_path ASC LIMIT ?`,
				params: [agentId, source.id, ARTIFACT_KINDS[resource], after, STALE_SCAN_BATCH],
				result: "all",
			},
			{ operation: "sources.gohighlevel.scan_stale", lane: "read", deadlineMs: 5_000 },
		);
		for (const row of rows) {
			if (!seenPaths.has(row.source_path)) stale.push({ rowid: row.rowid, sourcePath: row.source_path });
		}
		if (rows.length < STALE_SCAN_BATCH) break;
		after = rows[rows.length - 1]?.source_path ?? after;
	}
	for (let start = 0; start < stale.length; start += PURGE_BATCH) {
		if (!shouldContinue()) return;
		const batch = stale.slice(start, start + PURGE_BATCH);
		for (const row of batch) {
			await purgeSourceArtifactStructureAsync({ agentId, sourceId: source.id, sourcePath: row.sourcePath });
		}
		await dbOwnerTransaction(
			[
				ownerStatement(
					`UPDATE memory_artifacts SET is_deleted = 1, updated_at = ?
					 WHERE agent_id = ? AND source_id = ? AND rowid IN (${batch.map(() => "?").join(", ")})`,
					[syncStartedAt, agentId, source.id, ...batch.map((row) => row.rowid)],
				),
			],
			{
				operation: "sources.gohighlevel.purge_stale",
				lane: "write",
				deadlineMs: 30_000,
				estimatedWorkUnits: batch.length,
			},
		);
	}
}

async function purgeStaleFailureArtifacts(sourceId: string, agentId: string, syncStartedAt: string): Promise<void> {
	await dbOwnerTransaction(
		[
			ownerStatement(
				`UPDATE memory_artifacts SET is_deleted = 1, updated_at = ?
				 WHERE agent_id = ? AND source_id = ? AND source_kind = '${FAILURE_KIND}'
				   AND updated_at < ? AND COALESCE(is_deleted, 0) = 0`,
				[syncStartedAt, agentId, sourceId, syncStartedAt],
			),
		],
		{ operation: "sources.gohighlevel.purge_failures", lane: "write", deadlineMs: 5_000, estimatedWorkUnits: 1 },
	);
}

async function writeFailureArtifact(
	source: SignetSourceEntry,
	agentId: string,
	failure: SourceFailureState,
): Promise<void> {
	const fingerprint = failureFingerprint(failure);
	await indexExternalMemoryArtifact({
		agentId,
		harness: GOHIGHLEVEL_HARNESS,
		sourceId: source.id,
		sourceRoot: source.root,
		sourceExternalId: `failure:${fingerprint}`,
		sourcePath: `gohighlevel://sources/${source.id}/failures/${fingerprint}`,
		sourceKind: FAILURE_KIND,
		sourceMtimeMs: Date.parse(failure.failedAt) || Date.now(),
		capturedAt: failure.failedAt,
		content: `# GoHighLevel sync failed\n\n${failure.message}`,
		sourceMeta: { provider: GOHIGHLEVEL_PROVIDER_KIND, recoverable: failure.recoverable, ...failure.metadata },
	});
}

function failureFingerprint(failure: SourceFailureState): string {
	return createHash("sha256")
		.update(failure.message)
		.update("\0")
		.update(JSON.stringify(failure.metadata ?? {}))
		.digest("hex")
		.slice(0, 16);
}

function failureState(
	source: SignetSourceEntry,
	message: string,
	metadata?: Readonly<Record<string, unknown>>,
): SourceFailureState {
	return {
		sourceId: source.id,
		providerKind: GOHIGHLEVEL_PROVIDER_KIND,
		failedAt: new Date().toISOString(),
		recoverable: true,
		message,
		metadata,
	};
}

async function resolveToken(tokenRef: string): Promise<string> {
	try {
		return await getSecret(tokenRef);
	} catch (err) {
		throw new Error(`Failed to resolve GoHighLevel token ref '${tokenRef}': ${errorMessage(err)}`);
	}
}

function errorMessage(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}
