import { createHash } from "node:crypto";
import {
	type SignetSourceEntry,
	type SourceFailureState,
	type SourceProviderKind,
	type StripeSourceResourceType,
	parseStripeSettings,
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
	STRIPE_PAGE_SIZE,
	type StripeListPage,
	type StripeRecord,
	StripeRequestError,
	listStripePage,
} from "./stripe-source-fetch";

const STRIPE_PROVIDER_KIND: SourceProviderKind = "stripe";
const STRIPE_HARNESS = "stripe";
const FAILURE_KIND = "source_stripe_failure";
export const STRIPE_MAX_CONSECUTIVE_REQUEST_FAILURES = 5;
export const STRIPE_SYNC_DEADLINE_MS = 3 * 60 * 60 * 1000;
const STALE_SCAN_BATCH = 500;
const PURGE_BATCH = 100;

const ARTIFACT_KINDS: Readonly<Record<StripeSourceResourceType, string>> = {
	customers: "source_stripe_customer",
	subscriptions: "source_stripe_subscription",
	invoices: "source_stripe_invoice",
	charges: "source_stripe_charge",
};

let syncDeadlineMs = STRIPE_SYNC_DEADLINE_MS;

export function setStripeSyncDeadlineForTest(ms: number | null): void {
	syncDeadlineMs = ms ?? STRIPE_SYNC_DEADLINE_MS;
}

type StopReason = { readonly message: string; readonly haltsSync: boolean };

interface ResourceOutcome {
	readonly scanned: number;
	readonly indexed: number;
	readonly complete: boolean;
	readonly seenPaths: ReadonlySet<string>;
	readonly haltSync: string | null;
}

export const stripeSourceProvider: SourceProviderAdapter = {
	kind: STRIPE_PROVIDER_KIND,
	sync: syncStripeSource,
	purge: async (source, agentId) => await purgeSourceOwnedRows({ sourceId: source.id, agentId }),
};

async function syncStripeSource(context: SourceProviderSyncContext): Promise<SourceProviderSyncResult> {
	const source = context.source;
	const agentId = context.agentId;
	const settings = parseStripeSettings(source.providerSettings);
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
					failureState(source, `Stripe stale ${resource} purge failed: ${errorMessage(err)}`, {
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
	readonly resource: StripeSourceResourceType;
	readonly maxItems: number;
	readonly deadline: number;
	readonly failures: SourceFailureState[];
	readonly scannedBefore: number;
	readonly indexedBefore: number;
}): Promise<ResourceOutcome> {
	const { context, token, resource, maxItems, deadline, failures } = input;
	const source = context.source;
	const agentId = context.agentId;
	const seenPaths = new Set<string>();
	const yielder = yieldEvery(5);
	let scanned = 0;
	let indexed = 0;
	let unparsed = 0;
	let cursor: string | null = null;
	let reachedEnd = false;
	let stop: StopReason | null = null;
	let consecutiveRequestFailures = 0;

	while (!reachedEnd && stop === null && scanned < maxItems && context.shouldContinue()) {
		if (Date.now() > deadline) {
			stop = {
				message: `Stripe sync reached its ${Math.round(syncDeadlineMs / 60_000)}-minute deadline`,
				haltsSync: true,
			};
			break;
		}
		let page: StripeListPage;
		try {
			page = await listStripePage(token, resource, cursor, Math.min(STRIPE_PAGE_SIZE, maxItems - scanned), () =>
				context.shouldContinue(),
			);
			consecutiveRequestFailures = 0;
		} catch (err) {
			if (!context.shouldContinue()) break;
			const kind = requestOutcome(err);
			if (kind === "unauthorized") {
				stop = { message: "Stripe rejected the API key", haltsSync: true };
			} else if (kind === "forbidden") {
				stop = { message: `The Stripe API key cannot read ${resource}: ${errorMessage(err)}`, haltsSync: false };
			} else if (kind === "transient") {
				consecutiveRequestFailures++;
				if (consecutiveRequestFailures < STRIPE_MAX_CONSECUTIVE_REQUEST_FAILURES) continue;
				stop = {
					message: `Stopped after ${consecutiveRequestFailures} consecutive Stripe request failures: ${errorMessage(err)}`,
					haltsSync: true,
				};
			} else {
				stop = { message: `Stripe ${resource} list failed: ${errorMessage(err)}`, haltsSync: false };
			}
			break;
		}
		unparsed += page.unparsed;
		let versions: ReadonlyMap<string, string | null>;
		try {
			versions = await readIndexedContentHashes(
				source.id,
				agentId,
				ARTIFACT_KINDS[resource],
				page.records.map((record) => recordPath(source, record)),
			);
		} catch (err) {
			stop = { message: `Stripe index state read failed: ${errorMessage(err)}`, haltsSync: true };
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
			const content = recordContent(record);
			const hash = contentHash(content);
			if (versions.get(path) === hash) {
				indexed++;
			} else {
				try {
					await writeRecordArtifact(source, agentId, record, content, hash);
					indexed++;
				} catch (err) {
					failures.push(
						failureState(source, `Stripe ${record.resource} indexing failed for ${record.id}: ${errorMessage(err)}`, {
							phase: "index",
							resource,
							objectId: record.id,
						}),
					);
				}
			}
			scanned++;
			await yielder();
		}
		if (!page.hasMore) reachedEnd = true;
		else if (page.lastId === null) {
			stop = {
				message: `Stripe returned a ${resource} page with no cursor before the end of the list`,
				haltsSync: false,
			};
		} else cursor = page.lastId;
	}

	if (unparsed > 0) {
		failures.push(
			failureState(source, `Skipped ${unparsed} Stripe ${resource} object(s) with an unexpected shape`, {
				phase: "parse",
				resource,
				unparsed,
			}),
		);
	}
	const capped = !reachedEnd && stop === null && scanned >= maxItems;
	if (capped) {
		logger.warn("stripe-source", "Stripe enumeration hit configured cap", {
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
	if (!(err instanceof StripeRequestError)) return "failed";
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
	record: StripeRecord,
	content: string,
	hash: string,
): Promise<void> {
	const sourcePath = recordPath(source, record);
	const sourceParentPath = collectionPath(source, record.resource);
	const sourceKind = ARTIFACT_KINDS[record.resource];
	const createdMs = record.createdMs ?? Date.now();
	await indexExternalMemoryArtifact({
		agentId,
		harness: STRIPE_HARNESS,
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
			provider: STRIPE_PROVIDER_KIND,
			resource: record.resource,
			objectId: record.id,
			customerId: record.customerId,
			livemode: record.livemode,
			contentHash: hash,
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

export function recordContent(record: StripeRecord): string {
	const lines = [`# ${record.title}`, "", `Stripe ${singular(record.resource)}: ${record.id}`];
	if (!record.livemode) lines.push("Mode: test");
	if (record.fields.length > 0) {
		lines.push("");
		for (const entry of record.fields) lines.push(`- ${entry.name}: ${entry.value}`);
	}
	return lines.join("\n");
}

function singular(resource: StripeSourceResourceType): string {
	return resource.slice(0, -1);
}

function recordPath(source: SignetSourceEntry, record: StripeRecord): string {
	return `${collectionPath(source, record.resource)}/${record.id}`;
}

function collectionPath(source: SignetSourceEntry, resource: StripeSourceResourceType): string {
	return `stripe://sources/${source.id}/${resource}`;
}

async function readIndexedContentHashes(
	sourceId: string,
	agentId: string,
	sourceKind: string,
	paths: readonly string[],
): Promise<ReadonlyMap<string, string | null>> {
	const hashes = new Map<string, string | null>();
	if (paths.length === 0) return hashes;
	const rows = await dbOwnerQuery<Array<{ readonly source_path: string; readonly content_hash: string | null }>>(
		{
			sql: `SELECT source_path, json_extract(source_meta_json, '$.contentHash') AS content_hash
			 FROM memory_artifacts
			 WHERE agent_id = ? AND source_id = ? AND source_kind = ?
			   AND COALESCE(is_deleted, 0) = 0
			   AND source_path IN (${paths.map(() => "?").join(", ")})`,
			params: [agentId, sourceId, sourceKind, ...paths],
			result: "all",
		},
		{ operation: "sources.stripe.read_versions", lane: "read", deadlineMs: 5_000 },
	);
	for (const row of rows) {
		hashes.set(row.source_path, typeof row.content_hash === "string" ? row.content_hash : null);
	}
	return hashes;
}

async function purgeStaleRecords(
	source: SignetSourceEntry,
	agentId: string,
	resource: StripeSourceResourceType,
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
			{ operation: "sources.stripe.scan_stale", lane: "read", deadlineMs: 5_000 },
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
			{ operation: "sources.stripe.purge_stale", lane: "write", deadlineMs: 30_000, estimatedWorkUnits: batch.length },
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
		{ operation: "sources.stripe.purge_failures", lane: "write", deadlineMs: 5_000, estimatedWorkUnits: 1 },
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
		harness: STRIPE_HARNESS,
		sourceId: source.id,
		sourceRoot: source.root,
		sourceExternalId: `failure:${fingerprint}`,
		sourcePath: `stripe://sources/${source.id}/failures/${fingerprint}`,
		sourceKind: FAILURE_KIND,
		sourceMtimeMs: Date.parse(failure.failedAt) || Date.now(),
		capturedAt: failure.failedAt,
		content: `# Stripe sync failed\n\n${failure.message}`,
		sourceMeta: { provider: STRIPE_PROVIDER_KIND, recoverable: failure.recoverable, ...failure.metadata },
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
		providerKind: STRIPE_PROVIDER_KIND,
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
		throw new Error(`Failed to resolve Stripe API key ref '${tokenRef}': ${errorMessage(err)}`);
	}
}

function errorMessage(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}
