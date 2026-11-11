import { createHash } from "node:crypto";
import {
	type SignetSourceEntry,
	type SourceFailureState,
	type SourceProviderKind,
	parseGmailSettings,
} from "@signet/core";
import { yieldEvery } from "./async-yield";
import { dbOwnerQuery, dbOwnerTransaction, ownerStatement } from "./db-owner-runtime";
import {
	GMAIL_LIST_PAGE_SIZE,
	GMAIL_MAX_BODY_CHARS,
	type GmailAccessToken,
	type GmailCredentials,
	type GmailMessage,
	type GmailMessageList,
	type GmailMessageLookup,
	GmailRequestError,
	fetchGmailAccessToken,
	fetchGmailMessage,
	gmailTokenIsFresh,
	listGmailMessages,
	parseGmailCredentials,
} from "./gmail-source-fetch";
import { logger } from "./logger";
import { indexExternalMemoryArtifact } from "./memory-lineage";
import { getSecret } from "./secrets";
import { indexSourceArtifactStructureAsync, purgeSourceArtifactStructureAsync } from "./source-artifact-graph";
import type { SourceProviderAdapter, SourceProviderSyncContext, SourceProviderSyncResult } from "./source-providers";
import { purgeSourceOwnedRows } from "./source-purge";

const GMAIL_PROVIDER_KIND: SourceProviderKind = "gmail";
const GMAIL_HARNESS = "gmail";
const MESSAGE_KIND = "source_gmail_message";
const FAILURE_KIND = "source_gmail_failure";
export const GMAIL_MAX_CONSECUTIVE_REQUEST_FAILURES = 5;
export const GMAIL_SYNC_DEADLINE_MS = 3 * 60 * 60 * 1000;
const STALE_SCAN_BATCH = 500;
const PURGE_BATCH = 100;

let syncDeadlineMs = GMAIL_SYNC_DEADLINE_MS;

export function setGmailSyncDeadlineForTest(ms: number | null): void {
	syncDeadlineMs = ms ?? GMAIL_SYNC_DEADLINE_MS;
}

type RequestOutcome = "unauthorized" | "transient" | "failed";

export const gmailSourceProvider: SourceProviderAdapter = {
	kind: GMAIL_PROVIDER_KIND,
	sync: syncGmailSource,
	purge: async (source, agentId) => await purgeSourceOwnedRows({ sourceId: source.id, agentId }),
};

class GmailSession {
	private access: GmailAccessToken | null = null;

	constructor(
		private readonly credentials: GmailCredentials,
		private readonly shouldContinue: () => boolean,
	) {}

	async call<T>(request: (token: string) => Promise<T>): Promise<T> {
		try {
			return await request(await this.token(false));
		} catch (err) {
			if (!(err instanceof GmailRequestError) || err.status !== 401) throw err;
			return await request(await this.token(true));
		}
	}

	private async token(forceRefresh: boolean): Promise<string> {
		if (forceRefresh || !gmailTokenIsFresh(this.access)) {
			this.access = await fetchGmailAccessToken(this.credentials, this.shouldContinue);
		}
		return this.access.token;
	}
}

async function syncGmailSource(context: SourceProviderSyncContext): Promise<SourceProviderSyncResult> {
	const source = context.source;
	const agentId = context.agentId;
	const settings = parseGmailSettings(source.providerSettings);
	const syncStartedAt = new Date().toISOString();
	const session = new GmailSession(parseGmailCredentials(await resolveSecret(settings.tokenRef)), () =>
		context.shouldContinue(),
	);
	const failures: SourceFailureState[] = [];
	const deadline = Date.now() + syncDeadlineMs;
	const seenPaths = new Set<string>();
	const yielder = yieldEvery(5);
	let indexed = 0;
	let scanned = 0;
	let unfetched = 0;
	let pageToken: string | null = null;
	let reachedEnd = false;
	let stopReason: string | null = null;
	let consecutiveRequestFailures = 0;
	context.onProgress?.({ scanned, total: scanned, indexed, currentPath: source.root });

	const stopFor = (err: unknown, what: string): string | null => {
		const outcome = requestOutcome(err);
		if (outcome === "unauthorized") return `Google rejected the Gmail credentials: ${errorMessage(err)}`;
		if (outcome === "transient") {
			consecutiveRequestFailures++;
			if (consecutiveRequestFailures < GMAIL_MAX_CONSECUTIVE_REQUEST_FAILURES) return null;
			return `Stopped after ${consecutiveRequestFailures} consecutive Gmail request failures: ${errorMessage(err)}`;
		}
		return `${what} failed: ${errorMessage(err)}`;
	};

	while (!reachedEnd && stopReason === null && scanned < settings.maxMessages && context.shouldContinue()) {
		if (Date.now() > deadline) {
			stopReason = `Gmail sync reached its ${Math.round(syncDeadlineMs / 60_000)}-minute deadline`;
			break;
		}
		let page: GmailMessageList;
		try {
			const cursor = pageToken;
			const limit = Math.min(GMAIL_LIST_PAGE_SIZE, settings.maxMessages - scanned);
			page = await session.call((token) =>
				listGmailMessages(token, settings.query, cursor, limit, () => context.shouldContinue()),
			);
			consecutiveRequestFailures = 0;
		} catch (err) {
			if (!context.shouldContinue()) break;
			stopReason = stopFor(err, "Gmail message list");
			if (stopReason !== null) break;
			continue;
		}
		let known: ReadonlySet<string>;
		try {
			known = await readIndexedPaths(
				source.id,
				agentId,
				page.messages.map((message) => messagePath(source, message.id)),
			);
		} catch (err) {
			stopReason = `Gmail index state read failed: ${errorMessage(err)}`;
			break;
		}
		for (const ref of page.messages) {
			if (!context.shouldContinue() || stopReason !== null) break;
			if (Date.now() > deadline) {
				stopReason = `Gmail sync reached its ${Math.round(syncDeadlineMs / 60_000)}-minute deadline`;
				break;
			}
			const path = messagePath(source, ref.id);
			seenPaths.add(path);
			context.onProgress?.({ scanned, total: scanned, indexed, currentPath: path });
			if (known.has(path)) {
				indexed++;
			} else {
				let lookup: GmailMessageLookup | null = null;
				try {
					lookup = await session.call((token) => fetchGmailMessage(token, ref.id, () => context.shouldContinue()));
					consecutiveRequestFailures = 0;
				} catch (err) {
					if (!context.shouldContinue()) break;
					unfetched++;
					const outcome = requestOutcome(err);
					if (outcome === "failed") {
						failures.push(
							failureState(source, `Gmail message ${ref.id} could not be fetched: ${errorMessage(err)}`, {
								phase: "message",
								messageId: ref.id,
							}),
						);
					} else {
						stopReason = stopFor(err, "Gmail message fetch");
						if (stopReason !== null) break;
					}
				}
				if (lookup?.status === "gone") seenPaths.delete(path);
				if (lookup?.status === "found") {
					try {
						await writeMessageArtifact(source, agentId, lookup.message);
						indexed++;
					} catch (err) {
						failures.push(
							failureState(source, `Gmail message ${ref.id} indexing failed: ${errorMessage(err)}`, {
								phase: "index",
								messageId: ref.id,
							}),
						);
					}
				}
			}
			scanned++;
			await yielder();
		}
		if (stopReason !== null) break;
		if (page.nextPageToken === null) reachedEnd = true;
		else pageToken = page.nextPageToken;
	}

	if (unfetched > 0 && stopReason === null && context.shouldContinue()) {
		failures.push(
			failureState(source, `${unfetched} Gmail message(s) could not be fetched and will be retried on the next sync`, {
				phase: "message",
				unfetched,
			}),
		);
	}
	const capped = !reachedEnd && stopReason === null && scanned >= settings.maxMessages;
	if (capped) {
		logger.warn("gmail-source", "Gmail enumeration hit configured cap", {
			sourceId: source.id,
			maxMessages: settings.maxMessages,
		});
	}
	if (stopReason !== null && context.shouldContinue()) {
		failures.push(
			failureState(source, `${stopReason}; removed messages were not purged`, {
				phase: "sync",
				scanned,
				unfetched,
			}),
		);
	}
	if (reachedEnd && stopReason === null && context.shouldContinue()) {
		try {
			await purgeStaleMessages(source, agentId, seenPaths, syncStartedAt, () => context.shouldContinue());
		} catch (err) {
			failures.push(failureState(source, `Gmail stale message purge failed: ${errorMessage(err)}`, { phase: "purge" }));
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

function requestOutcome(err: unknown): RequestOutcome {
	if (!(err instanceof GmailRequestError)) return "failed";
	if (
		err.status === 401 ||
		err.code === "invalid_grant" ||
		err.code === "invalid_client" ||
		err.code === "invalid_credentials"
	)
		return "unauthorized";
	if (err.code !== "cancelled" && (err.retryable || err.status === 0)) return "transient";
	return "failed";
}

async function writeMessageArtifact(source: SignetSourceEntry, agentId: string, message: GmailMessage): Promise<void> {
	const sourcePath = messagePath(source, message.id);
	const sourceParentPath = message.threadId ? `gmail://sources/${source.id}/threads/${message.threadId}` : source.root;
	const content = messageContent(message);
	const sentAtMs = message.sentAtMs ?? Date.now();
	await indexExternalMemoryArtifact({
		agentId,
		harness: GMAIL_HARNESS,
		sourceId: source.id,
		sourceRoot: source.root,
		sourceExternalId: message.id,
		sourceParentPath,
		sourcePath,
		sourceKind: MESSAGE_KIND,
		sourceMtimeMs: sentAtMs,
		capturedAt: new Date(sentAtMs).toISOString(),
		content,
		sourceMeta: {
			provider: GMAIL_PROVIDER_KIND,
			messageId: message.id,
			threadId: message.threadId,
			attachments: message.attachmentCount,
			clipped: message.bodyClipped,
		},
	});
	await indexSourceArtifactStructureAsync({
		agentId,
		sourceId: source.id,
		sourceKind: MESSAGE_KIND,
		sourceRoot: source.root,
		sourceParentPath,
		sourcePath,
		displayName: message.subject || "(no subject)",
		content,
	});
}

export function messageContent(message: GmailMessage): string {
	const lines = [`# ${message.subject || "(no subject)"}`, "", `Gmail message: ${message.id}`];
	if (message.from) lines.push(`- From: ${message.from}`);
	if (message.to) lines.push(`- To: ${message.to}`);
	if (message.cc) lines.push(`- Cc: ${message.cc}`);
	if (message.sentAtMs !== null) lines.push(`- Date: ${new Date(message.sentAtMs).toISOString()}`);
	if (message.threadId) lines.push(`- Thread: ${message.threadId}`);
	if (message.attachmentCount > 0) {
		const hidden = message.attachmentCount - message.attachmentNames.length;
		lines.push(`- Attachments: ${message.attachmentNames.join(", ")}${hidden > 0 ? ` and ${hidden} more` : ""}`);
	}
	if (message.body) lines.push("", message.body);
	if (message.bodyClipped) {
		lines.push(
			"",
			`> Signet stored the first ${GMAIL_MAX_BODY_CHARS.toLocaleString("en-US")} characters of this message.`,
		);
	}
	return lines.join("\n");
}

function messagePath(source: SignetSourceEntry, messageId: string): string {
	return `gmail://sources/${source.id}/messages/${messageId}`;
}

async function readIndexedPaths(
	sourceId: string,
	agentId: string,
	paths: readonly string[],
): Promise<ReadonlySet<string>> {
	const known = new Set<string>();
	for (let start = 0; start < paths.length; start += PURGE_BATCH) {
		const batch = paths.slice(start, start + PURGE_BATCH);
		const rows = await dbOwnerQuery<Array<{ readonly source_path: string }>>(
			{
				sql: `SELECT source_path FROM memory_artifacts
				 WHERE agent_id = ? AND source_id = ? AND source_kind = '${MESSAGE_KIND}'
				   AND COALESCE(is_deleted, 0) = 0
				   AND source_path IN (${batch.map(() => "?").join(", ")})`,
				params: [agentId, sourceId, ...batch],
				result: "all",
			},
			{ operation: "sources.gmail.read_indexed", lane: "read", deadlineMs: 30_000 },
		);
		for (const row of rows) known.add(row.source_path);
	}
	return known;
}

async function purgeStaleMessages(
	source: SignetSourceEntry,
	agentId: string,
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
				 WHERE agent_id = ? AND source_id = ? AND source_kind = '${MESSAGE_KIND}'
				   AND COALESCE(is_deleted, 0) = 0 AND source_path > ?
				 ORDER BY source_path ASC LIMIT ?`,
				params: [agentId, source.id, after, STALE_SCAN_BATCH],
				result: "all",
			},
			{ operation: "sources.gmail.scan_stale", lane: "read", deadlineMs: 30_000 },
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
			{ operation: "sources.gmail.purge_stale", lane: "write", deadlineMs: 30_000, estimatedWorkUnits: batch.length },
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
		{ operation: "sources.gmail.purge_failures", lane: "write", deadlineMs: 30_000, estimatedWorkUnits: 1 },
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
		harness: GMAIL_HARNESS,
		sourceId: source.id,
		sourceRoot: source.root,
		sourceExternalId: `failure:${fingerprint}`,
		sourcePath: `gmail://sources/${source.id}/failures/${fingerprint}`,
		sourceKind: FAILURE_KIND,
		sourceMtimeMs: Date.parse(failure.failedAt) || Date.now(),
		capturedAt: failure.failedAt,
		content: `# Gmail sync failed\n\n${failure.message}`,
		sourceMeta: { provider: GMAIL_PROVIDER_KIND, recoverable: failure.recoverable, ...failure.metadata },
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
		providerKind: GMAIL_PROVIDER_KIND,
		failedAt: new Date().toISOString(),
		recoverable: true,
		message,
		metadata,
	};
}

async function resolveSecret(tokenRef: string): Promise<string> {
	try {
		return await getSecret(tokenRef);
	} catch (err) {
		throw new Error(`Failed to resolve Gmail credentials ref '${tokenRef}': ${errorMessage(err)}`);
	}
}

function errorMessage(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}
