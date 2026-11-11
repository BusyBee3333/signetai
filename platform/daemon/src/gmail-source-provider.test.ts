import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type SignetSourceEntry, addGmailSource } from "@signet/core";
import type { SecretKeyringAdapter } from "@signet/core";
import { closeDbAccessor, getDbAccessor, initDbAccessor } from "./db-accessor";
import {
	GMAIL_MAX_BODY_CHARS,
	GOOGLE_TOKEN_URL,
	parseGmailCredentials,
	parseGmailMessage,
	setGmailSleepForTest,
} from "./gmail-source-fetch";
import { gmailSourceProvider, setGmailSyncDeadlineForTest } from "./gmail-source-provider";
import { putSecret, setSecretKeyringAdapterForTests } from "./secrets";
import { htmlToText } from "./source-html-text";

const originalFetch = globalThis.fetch;
const CLIENT_SECRET = "client-secret-value";
const REFRESH_TOKEN = "refresh-token-value";
const CREDENTIALS = JSON.stringify({
	type: "authorized_user",
	client_id: "client-id.apps.googleusercontent.com",
	client_secret: CLIENT_SECRET,
	refresh_token: REFRESH_TOKEN,
});

interface RecordedRequest {
	readonly method: string;
	readonly url: string;
	readonly path: string;
	readonly query: URLSearchParams;
	readonly headers: Headers;
	readonly body: string;
}

type Route = (request: RecordedRequest) => Response | Promise<Response>;

function memoryKeyring(): SecretKeyringAdapter {
	let stored: string | undefined;
	const read = async () =>
		stored === undefined ? { state: "missing" as const } : { state: "found" as const, value: stored };
	return {
		platform: "test",
		service: "ai.signet.secrets",
		account: "gmail-source-test",
		get: read,
		getStatus: read,
		async set(value: string) {
			stored = value;
			return { state: "found" as const, value };
		},
	};
}

function stubGoogle(route: Route): RecordedRequest[] {
	const requests: RecordedRequest[] = [];
	let tokens = 0;
	globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
		const url = new URL(String(input));
		const request: RecordedRequest = {
			method: init?.method ?? "GET",
			url: `${url.origin}${url.pathname}`,
			path: url.pathname.replace("/gmail/v1/users/me", ""),
			query: url.searchParams,
			headers: new Headers(init?.headers),
			body: typeof init?.body === "string" ? init.body : "",
		};
		requests.push(request);
		if (request.url === GOOGLE_TOKEN_URL) {
			const routed = await route(request);
			if (routed.status !== 599) return routed;
			tokens++;
			return Response.json({ access_token: `access-${tokens}`, expires_in: 3600, token_type: "Bearer" });
		}
		return await route(request);
	}) as typeof fetch;
	return requests;
}

const USE_DEFAULT_TOKEN = (): Response => new Response(null, { status: 599 });

function b64(value: string): string {
	return Buffer.from(value, "utf8").toString("base64url");
}

function message(id: string, subject: string, bodyText: string, extra: Readonly<Record<string, unknown>> = {}) {
	return {
		id,
		threadId: `t-${id}`,
		internalDate: "1767225600000",
		payload: {
			mimeType: "multipart/mixed",
			headers: [
				{ name: "Subject", value: subject },
				{ name: "From", value: "Owner <owner@example.com>" },
				{ name: "To", value: "billing@example.com" },
			],
			parts: [
				{ mimeType: "text/plain", filename: "", body: { data: b64(bodyText) } },
				{ mimeType: "application/pdf", filename: "agreement.pdf", body: { attachmentId: "att-1", size: 1234 } },
			],
		},
		...extra,
	};
}

function mailbox(messages: readonly ReturnType<typeof message>[], pageSize = 500): Route {
	return (request) => {
		if (request.url === GOOGLE_TOKEN_URL) return USE_DEFAULT_TOKEN();
		if (request.path === "/messages") {
			const start = Number(request.query.get("pageToken") ?? "0");
			const slice = messages.slice(start, start + pageSize);
			const next = start + pageSize < messages.length ? String(start + pageSize) : undefined;
			return Response.json({
				messages: slice.map((entry) => ({ id: entry.id, threadId: entry.threadId })),
				...(next ? { nextPageToken: next } : {}),
				resultSizeEstimate: messages.length,
			});
		}
		const id = request.path.match(/^\/messages\/([^/]+)$/)?.[1];
		const found = messages.find((entry) => entry.id === id);
		return found ? Response.json(found) : googleError(404, "notFound");
	};
}

function googleError(status: number, reason: string, headers: Record<string, string> = {}): Response {
	return Response.json(
		{ error: { code: status, message: `${reason} message`, errors: [{ reason }], status: reason } },
		{ status, headers },
	);
}

describe("gmail-source-provider", () => {
	let dir = "";
	let previousSignetPath: string | undefined;
	let sleeps: number[] = [];

	beforeEach(async () => {
		dir = mkdtempSync(join(tmpdir(), "signet-gmail-source-"));
		previousSignetPath = process.env.SIGNET_PATH;
		process.env.SIGNET_PATH = dir;
		mkdirSync(join(dir, "memory"), { recursive: true });
		closeDbAccessor();
		initDbAccessor(join(dir, "memory", "memories.db"));
		setSecretKeyringAdapterForTests(memoryKeyring());
		await putSecret("GMAIL_OAUTH", CREDENTIALS);
		sleeps = [];
		setGmailSleepForTest(async (ms) => {
			sleeps.push(ms);
		});
	});

	afterEach(async () => {
		globalThis.fetch = originalFetch;
		setGmailSleepForTest(null);
		setGmailSyncDeadlineForTest(null);
		setSecretKeyringAdapterForTests(null);
		await closeDbAccessor();
		if (previousSignetPath === undefined) Reflect.deleteProperty(process.env, "SIGNET_PATH");
		else process.env.SIGNET_PATH = previousSignetPath;
		rmSync(dir, { recursive: true, force: true });
	});

	function addSource(input: { query?: string; maxMessages?: number } = {}): SignetSourceEntry {
		const added = addGmailSource({ tokenRef: "GMAIL_OAUTH", ...input }, dir);
		if (added.ok === false) throw new Error(added.error);
		return added.source;
	}

	async function sync(source: SignetSourceEntry, shouldContinue: () => boolean = () => true) {
		const result = await gmailSourceProvider.sync?.({ source, agentsDir: dir, agentId: "default", shouldContinue });
		if (!result) throw new Error("Gmail provider has no sync");
		return result;
	}

	function messageIds(sourceId: string): Array<string | null> {
		return liveRows(sourceId)
			.filter((row) => row.source_kind === "source_gmail_message")
			.map((row) => row.source_external_id);
	}

	it("exchanges the refresh token and indexes messages as attributable source artifacts", async () => {
		const source = addSource({ query: "from:owner@example.com" });
		const requests = stubGoogle(mailbox([message("m1", "Cancel my plan", "Please cancel at the end of the term.")]));

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(result).toMatchObject({ indexed: 1, scanned: 1, total: 1 });
		const tokenRequest = requests[0];
		expect(tokenRequest?.url).toBe(GOOGLE_TOKEN_URL);
		expect(tokenRequest?.method).toBe("POST");
		expect(new URLSearchParams(tokenRequest?.body).get("grant_type")).toBe("refresh_token");
		expect(new URLSearchParams(tokenRequest?.body).get("refresh_token")).toBe(REFRESH_TOKEN);
		const listRequest = requests[1];
		expect(listRequest?.path).toBe("/messages");
		expect(listRequest?.headers.get("authorization")).toBe("Bearer access-1");
		expect(listRequest?.query.get("q")).toBe("from:owner@example.com");
		expect(listRequest?.query.get("maxResults")).toBe("500");
		expect(requests[2]?.query.get("format")).toBe("full");

		const [row] = liveRows(source.id);
		expect(row?.source_kind).toBe("source_gmail_message");
		expect(row?.source_path).toBe(`gmail://sources/${source.id}/messages/m1`);
		expect(row?.source_parent_path).toBe(`gmail://sources/${source.id}/threads/t-m1`);
		expect(row?.content).toContain("# Cancel my plan");
		expect(row?.content).toContain("- From: Owner <owner@example.com>");
		expect(row?.content).toContain("- Date: 2026-01-01T00:00:00.000Z");
		expect(row?.content).toContain("- Attachments: agreement.pdf");
		expect(row?.content).toContain("Please cancel at the end of the term.");
		expect(JSON.parse(row?.source_meta_json ?? "{}")).toMatchObject({
			provider: "gmail",
			messageId: "m1",
			threadId: "t-m1",
			attachments: 1,
			clipped: false,
		});
		const stored = allRows(source.id)
			.map((entry) => `${entry.content}\n${entry.source_meta_json ?? ""}`)
			.join("\n");
		for (const secret of [CLIENT_SECRET, REFRESH_TOKEN, "access-1", "att-1"]) expect(stored).not.toContain(secret);
	});

	it("follows page tokens to the end of the mailbox", async () => {
		const source = addSource();
		const requests = stubGoogle(
			mailbox([message("m3", "C", "c"), message("m2", "B", "b"), message("m1", "A", "a")], 2),
		);

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(result).toMatchObject({ indexed: 3, scanned: 3 });
		expect(
			requests.filter((request) => request.path === "/messages").map((request) => request.query.get("pageToken")),
		).toEqual([null, "2"]);
		expect(messageIds(source.id)).toEqual(["m1", "m2", "m3"]);
	});

	it("does not refetch indexed messages and purges messages gone after a complete enumeration", async () => {
		const source = addSource();
		stubGoogle(mailbox([message("m2", "B", "b"), message("m1", "A", "a")]));
		await sync(source);

		const requests = stubGoogle(mailbox([message("m3", "C", "c"), message("m1", "A", "a")]));
		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(requests.filter((request) => /^\/messages\/./.test(request.path)).map((request) => request.path)).toEqual([
			"/messages/m3",
		]);
		expect(messageIds(source.id)).toEqual(["m1", "m3"]);
		expect(allRows(source.id).find((row) => row.source_external_id === "m2")?.is_deleted).toBe(1);
	});

	it("keeps existing messages when the configured cap ends enumeration early", async () => {
		const source = addSource();
		stubGoogle(mailbox([message("m2", "B", "b"), message("m1", "A", "a")]));
		await sync(source);

		const capped = addSource({ maxMessages: 1 });
		const requests = stubGoogle(mailbox([message("m2", "B", "b"), message("m1", "A", "a")], 1));
		const result = await sync(capped);

		expect(requests.find((request) => request.path === "/messages")?.query.get("maxResults")).toBe("1");
		expect(result).toMatchObject({ scanned: 1 });
		expect(messageIds(source.id)).toEqual(["m1", "m2"]);
	});

	it("refreshes the access token once when Gmail answers 401", async () => {
		const source = addSource();
		const inner = mailbox([message("m1", "A", "a")]);
		let rejected = false;
		const requests = stubGoogle((request) => {
			if (request.path === "/messages" && !rejected) {
				rejected = true;
				return googleError(401, "authError");
			}
			return inner(request);
		});

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(requests.filter((request) => request.url === GOOGLE_TOKEN_URL)).toHaveLength(2);
		expect(messageIds(source.id)).toEqual(["m1"]);
	});

	it("stops and keeps existing messages when Google rejects the credentials", async () => {
		const source = addSource();
		stubGoogle(mailbox([message("m1", "A", "a")]));
		await sync(source);

		stubGoogle(() =>
			Response.json(
				{ error: "invalid_grant", error_description: "Token has been expired or revoked." },
				{ status: 400 },
			),
		);
		const result = await sync(source);

		expect(result.failures.map((failure) => failure.message)).toEqual([
			expect.stringContaining("Google rejected the Gmail credentials"),
		]);
		expect(result.failures[0]?.message).toContain("removed messages were not purged");
		expect(messageIds(source.id)).toEqual(["m1"]);
		expect(allRows(source.id).every((row) => !row.content.includes(REFRESH_TOKEN))).toBe(true);
	});

	it("skips a message deleted between listing and fetching without reporting a failure", async () => {
		const source = addSource();
		const inner = mailbox([message("m2", "B", "b"), message("m1", "A", "a")]);
		stubGoogle((request) => (request.path === "/messages/m2" ? googleError(404, "notFound") : inner(request)));

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(messageIds(source.id)).toEqual(["m1"]);
	});

	it("reports messages it could not fetch and indexes the rest", async () => {
		const source = addSource();
		const inner = mailbox([message("m2", "B", "b"), message("m1", "A", "a")]);
		stubGoogle((request) => (request.path === "/messages/m2" ? googleError(503, "backendError") : inner(request)));

		const result = await sync(source);

		expect(result.failures.map((failure) => failure.message)).toEqual([
			expect.stringContaining("1 Gmail message(s) could not be fetched"),
		]);
		expect(messageIds(source.id)).toEqual(["m1"]);
		expect(sleeps).toHaveLength(3);
	});

	it("honors Retry-After on rate-limited requests", async () => {
		const source = addSource();
		const inner = mailbox([message("m1", "A", "a")]);
		let limited = false;
		stubGoogle((request) => {
			if (request.path === "/messages" && !limited) {
				limited = true;
				return googleError(403, "userRateLimitExceeded", { "retry-after": "4" });
			}
			return inner(request);
		});

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(sleeps).toEqual([4_000]);
	});

	it("rejects a credentials secret that is not usable without echoing it", async () => {
		await putSecret("GMAIL_OAUTH", "not-json-secret-value");
		const source = addSource();
		const requests = stubGoogle(mailbox([]));

		await expect(sync(source)).rejects.toThrow("Gmail credentials secret is not JSON");
		await expect(sync(source)).rejects.not.toThrow("not-json-secret-value");
		expect(requests).toEqual([]);
	});

	it("stops without recording failures when the sync is cancelled", async () => {
		const source = addSource();
		let live = true;
		const inner = mailbox([message("m2", "B", "b"), message("m1", "A", "a")]);
		stubGoogle((request) => {
			if (request.path === "/messages/m2") live = false;
			return inner(request);
		});

		const result = await sync(source, () => live);

		expect(result.failures).toEqual([]);
		expect(allRows(source.id).some((row) => row.source_kind === "source_gmail_failure")).toBe(false);
		expect(messageIds(source.id)).not.toContain("m1");
	});
});

describe("gmail-source-fetch", () => {
	it("requires client_id, client_secret, and refresh_token", () => {
		expect(() => parseGmailCredentials(JSON.stringify({ client_id: "a", client_secret: "b" }))).toThrow(
			"must contain client_id, client_secret, and refresh_token",
		);
		expect(parseGmailCredentials(CREDENTIALS)).toEqual({
			clientId: "client-id.apps.googleusercontent.com",
			clientSecret: CLIENT_SECRET,
			refreshToken: REFRESH_TOKEN,
		});
	});

	it("falls back to HTML text, ignores attachment bodies, and clips long bodies", () => {
		const html = parseGmailMessage({
			id: "h1",
			payload: {
				mimeType: "text/html",
				headers: [{ name: "subject", value: "Hi" }],
				body: { data: b64("<style>p{}</style><p>Hello&nbsp;<b>there</b> &amp; welcome</p><script>x()</script>") },
			},
		});
		expect(html?.body).toBe("Hello there & welcome");
		const attachmentOnly = parseGmailMessage({
			id: "a1",
			payload: { parts: [{ mimeType: "text/plain", filename: "notes.txt", body: { data: b64("private file") } }] },
		});
		expect(attachmentOnly?.body).toBe("");
		expect(attachmentOnly?.attachmentNames).toEqual(["notes.txt"]);
		const long = parseGmailMessage({
			id: "l1",
			payload: { mimeType: "text/plain", body: { data: b64("x".repeat(GMAIL_MAX_BODY_CHARS + 10)) } },
		});
		expect(long?.body).toHaveLength(GMAIL_MAX_BODY_CHARS);
		expect(long?.bodyClipped).toBe(true);
		expect(parseGmailMessage({ payload: {} })).toBeNull();
	});

	it("converts simple HTML to text", () => {
		expect(htmlToText("<div>Line one</div><div>Line&nbsp;two<br>three</div>")).toBe("Line one\nLine two\nthree");
	});
});

interface ArtifactRow {
	source_kind: string;
	source_path: string;
	source_external_id: string | null;
	source_parent_path: string | null;
	source_meta_json: string | null;
	content: string;
	updated_at: string | null;
	is_deleted: number;
}

function allRows(sourceId: string): ArtifactRow[] {
	return getDbAccessor().withReadDb(
		(db) =>
			db
				.prepare(
					`SELECT source_kind, source_path, source_external_id, source_parent_path, source_meta_json, content,
					        updated_at, COALESCE(is_deleted, 0) AS is_deleted
					 FROM memory_artifacts
					 WHERE source_id = ?
					 ORDER BY source_path ASC`,
				)
				.all(sourceId) as ArtifactRow[],
	);
}

function liveRows(sourceId: string): ArtifactRow[] {
	return allRows(sourceId).filter((row) => row.is_deleted === 0);
}
