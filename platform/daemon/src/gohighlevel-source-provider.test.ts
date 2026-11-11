import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type SignetSourceEntry, addGoHighLevelSource } from "@signet/core";
import type { SecretKeyringAdapter } from "@signet/core";
import { closeDbAccessor, getDbAccessor, initDbAccessor } from "./db-accessor";
import { putSecret, setSecretKeyringAdapterForTests } from "./secrets";
import {
	GOHIGHLEVEL_API_VERSION,
	GOHIGHLEVEL_USER_AGENT,
	parseGoHighLevelRecord,
	setGoHighLevelSleepForTest,
} from "./gohighlevel-source-fetch";
import { goHighLevelSourceProvider, setGoHighLevelSyncDeadlineForTest } from "./gohighlevel-source-provider";

const originalFetch = globalThis.fetch;
const TOKEN = "gohighlevel-test-token-value";
const LOCATION = "LocAbc123456789";

interface RecordedRequest {
	readonly path: string;
	readonly query: URLSearchParams;
	readonly headers: Headers;
}

type Route = (request: RecordedRequest) => Response | Promise<Response>;

function memoryKeyring(): SecretKeyringAdapter {
	let stored: string | undefined;
	const read = async () =>
		stored === undefined ? { state: "missing" as const } : { state: "found" as const, value: stored };
	return {
		platform: "test",
		service: "ai.signet.secrets",
		account: "gohighlevel-source-test",
		get: read,
		getStatus: read,
		async set(value: string) {
			stored = value;
			return { state: "found" as const, value };
		},
	};
}

function stubGoHighLevel(route: Route): RecordedRequest[] {
	const requests: RecordedRequest[] = [];
	globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
		const url = new URL(String(input));
		const request: RecordedRequest = {
			path: url.pathname,
			query: url.searchParams,
			headers: new Headers(init?.headers),
		};
		requests.push(request);
		return await route(request);
	}) as typeof fetch;
	return requests;
}

function page(key: string, items: readonly unknown[], total: number, more = true): Response {
	const last = items[items.length - 1] as { id?: string } | undefined;
	return Response.json({
		[key]: items,
		meta:
			items.length > 0 && more
				? { total, startAfter: 1_700_000_000_000, startAfterId: last?.id ?? "", nextPage: 2 }
				: { total, nextPage: "" },
	});
}

function apiError(status: number, error: string, headers: Record<string, string> = {}): Response {
	return Response.json({ statusCode: status, error, message: `${error} message` }, { status, headers });
}

function contact(id: string, extra: Readonly<Record<string, unknown>> = {}): Record<string, unknown> {
	return {
		id,
		locationId: LOCATION,
		contactName: `Owner ${id}`,
		firstName: "Owner",
		lastName: id,
		companyName: `Gym ${id}`,
		email: `${id}@example.com`,
		phone: "+15550100",
		type: "customer",
		tags: ["blg", "active"],
		dnd: false,
		country: "US",
		address1: "1 Private Street",
		dateOfBirth: "1980-01-01",
		dateAdded: "2026-01-01T00:00:00.000Z",
		dateUpdated: "2026-02-01T00:00:00.000Z",
		customFields: [{ id: "cf_plan", value: "pro" }],
		...extra,
	};
}

const opportunity = {
	id: "opp_1",
	name: "Gym c1 renewal",
	status: "open",
	monetaryValue: 2970,
	pipelineId: "pipe_1",
	pipelineStageId: "stage_1",
	contactId: "c1",
	contact: { id: "c1", name: "Owner c1", companyName: "Gym c1", email: "c1@example.com" },
	createdAt: "2026-01-01T00:00:00.000Z",
	updatedAt: "2026-02-01T00:00:00.000Z",
};

function paged(key: string, pages: readonly (readonly unknown[])[], total: number): Route {
	return (request) => {
		const after = request.query.get("startAfterId");
		const index =
			after === null ? 0 : pages.findIndex((items) => (items[items.length - 1] as { id: string }).id === after) + 1;
		return page(key, pages[index] ?? [], total);
	};
}

function account(contacts: readonly unknown[] = [contact("c1")]): Route {
	return (request) =>
		request.path === "/contacts/"
			? paged("contacts", [contacts], contacts.length)(request)
			: paged("opportunities", [[opportunity]], 1)(request);
}

describe("gohighlevel-source-provider", () => {
	let dir = "";
	let previousSignetPath: string | undefined;
	let sleeps: number[] = [];

	beforeEach(async () => {
		dir = mkdtempSync(join(tmpdir(), "signet-gohighlevel-source-"));
		previousSignetPath = process.env.SIGNET_PATH;
		process.env.SIGNET_PATH = dir;
		mkdirSync(join(dir, "memory"), { recursive: true });
		closeDbAccessor();
		initDbAccessor(join(dir, "memory", "memories.db"));
		setSecretKeyringAdapterForTests(memoryKeyring());
		await putSecret("GHL_TOKEN", TOKEN);
		sleeps = [];
		setGoHighLevelSleepForTest(async (ms) => {
			sleeps.push(ms);
		});
	});

	afterEach(async () => {
		globalThis.fetch = originalFetch;
		setGoHighLevelSleepForTest(null);
		setGoHighLevelSyncDeadlineForTest(null);
		setSecretKeyringAdapterForTests(null);
		await closeDbAccessor();
		if (previousSignetPath === undefined) Reflect.deleteProperty(process.env, "SIGNET_PATH");
		else process.env.SIGNET_PATH = previousSignetPath;
		rmSync(dir, { recursive: true, force: true });
	});

	function addSource(input: { resourceTypes?: readonly string[]; maxItemsPerType?: number } = {}): SignetSourceEntry {
		const added = addGoHighLevelSource(
			{ tokenRef: "GHL_TOKEN", locationId: LOCATION, resourceTypes: ["contacts", "opportunities"], ...input },
			dir,
		);
		if (added.ok === false) throw new Error(added.error);
		return added.source;
	}

	async function sync(source: SignetSourceEntry, shouldContinue: () => boolean = () => true) {
		const result = await goHighLevelSourceProvider.sync?.({
			source,
			agentsDir: dir,
			agentId: "default",
			shouldContinue,
		});
		if (!result) throw new Error("GoHighLevel provider has no sync");
		return result;
	}

	function liveIds(sourceId: string, kind: string): Array<string | null> {
		return liveRows(sourceId)
			.filter((row) => row.source_kind === kind)
			.map((row) => row.source_external_id);
	}

	it("indexes contacts and opportunities as attributable source artifacts", async () => {
		const source = addSource();
		const requests = stubGoHighLevel(account());

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(result).toMatchObject({ indexed: 2, scanned: 2, total: 2 });
		for (const request of requests) {
			expect(request.headers.get("authorization")).toBe(`Bearer ${TOKEN}`);
			expect(request.headers.get("version")).toBe(GOHIGHLEVEL_API_VERSION);
			expect(request.headers.get("user-agent")).toBe(GOHIGHLEVEL_USER_AGENT);
			expect(request.query.get("limit")).toBe("100");
		}
		const contactsRequest = requests.find((request) => request.path === "/contacts/");
		const opportunitiesRequest = requests.find((request) => request.path === "/opportunities/search");
		expect(contactsRequest?.query.get("locationId")).toBe(LOCATION);
		expect(opportunitiesRequest?.query.get("location_id")).toBe(LOCATION);

		const rows = liveRows(source.id);
		expect(rows.map((row) => [row.source_kind, row.source_path])).toEqual([
			["source_gohighlevel_contact", `gohighlevel://sources/${source.id}/contacts/c1`],
			["source_gohighlevel_opportunity", `gohighlevel://sources/${source.id}/opportunities/opp_1`],
		]);
		const byId = new Map(rows.map((row) => [row.source_external_id, row]));
		expect(byId.get("c1")?.content).toContain("# Owner c1");
		expect(byId.get("c1")?.content).toContain("- Company: Gym c1");
		expect(byId.get("c1")?.content).toContain("- Tags: blg, active");
		expect(byId.get("c1")?.content).toContain("- Custom field cf_plan: pro");
		expect(byId.get("c1")?.content).not.toContain("Private Street");
		expect(byId.get("c1")?.content).not.toContain("1980-01-01");
		expect(byId.get("opp_1")?.content).toContain("# Gym c1 renewal");
		expect(byId.get("opp_1")?.content).toContain("- Value: 2970");
		expect(byId.get("opp_1")?.content).toContain("- Contact: c1");
		expect(JSON.parse(byId.get("opp_1")?.source_meta_json ?? "{}")).toMatchObject({
			provider: "gohighlevel",
			resource: "opportunities",
			objectId: "opp_1",
			contactId: "c1",
		});
		expect(allRows(source.id).every((row) => !`${row.content}${row.source_meta_json}`.includes(TOKEN))).toBe(true);
	});

	it("follows the startAfter cursor until GoHighLevel returns an empty page", async () => {
		const source = addSource({ resourceTypes: ["contacts"] });
		const requests = stubGoHighLevel(paged("contacts", [[contact("c3"), contact("c2")], [contact("c1")]], 3));

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(result).toMatchObject({ indexed: 3, scanned: 3 });
		expect(requests.map((request) => request.query.get("startAfterId"))).toEqual([null, "c2", "c1"]);
		expect(requests[1]?.query.get("startAfter")).toBe("1700000000000");
		expect(liveIds(source.id, "source_gohighlevel_contact")).toEqual(["c1", "c2", "c3"]);
	});

	it("leaves unchanged objects untouched and rewrites changed ones on the next sync", async () => {
		const source = addSource({ resourceTypes: ["contacts"] });
		stubGoHighLevel(account([contact("c1"), contact("c2")]));
		await sync(source);
		const before = new Map(allRows(source.id).map((row) => [row.source_external_id, row.updated_at]));
		await new Promise((resolve) => setTimeout(resolve, 5));

		stubGoHighLevel(account([contact("c1"), contact("c2", { tags: ["blg", "cancelled"] })]));
		await sync(source);

		const after = new Map(allRows(source.id).map((row) => [row.source_external_id, row]));
		expect(after.get("c1")?.updated_at).toBe(before.get("c1"));
		expect(after.get("c2")?.updated_at).not.toBe(before.get("c2"));
		expect(after.get("c2")?.content).toContain("- Tags: blg, cancelled");
	});

	it("purges objects that are gone after a complete enumeration", async () => {
		const source = addSource({ resourceTypes: ["contacts"] });
		stubGoHighLevel(account([contact("c1"), contact("c2")]));
		await sync(source);

		stubGoHighLevel(account([contact("c1")]));
		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(liveIds(source.id, "source_gohighlevel_contact")).toEqual(["c1"]);
	});

	it("keeps existing objects when the list ends before the reported total", async () => {
		const source = addSource({ resourceTypes: ["contacts"] });
		stubGoHighLevel(account([contact("c1"), contact("c2")]));
		await sync(source);

		stubGoHighLevel(paged("contacts", [[contact("c1")]], 2));
		const result = await sync(source);

		expect(result.failures.map((failure) => failure.message)).toEqual([
			expect.stringContaining("enumeration ended after 1 of 2 reported objects"),
		]);
		expect(liveIds(source.id, "source_gohighlevel_contact")).toEqual(["c1", "c2"]);
	});

	it("keeps existing objects when the configured cap ends enumeration early", async () => {
		const source = addSource({ resourceTypes: ["contacts"] });
		stubGoHighLevel(account([contact("c2"), contact("c1")]));
		await sync(source);

		const capped = addSource({ resourceTypes: ["contacts"], maxItemsPerType: 1 });
		const requests = stubGoHighLevel(paged("contacts", [[contact("c2")], [contact("c1")]], 2));
		const result = await sync(capped);

		expect(requests).toHaveLength(1);
		expect(requests[0]?.query.get("limit")).toBe("1");
		expect(result).toMatchObject({ scanned: 1 });
		expect(liveIds(source.id, "source_gohighlevel_contact")).toEqual(["c1", "c2"]);
	});

	it("keeps existing objects and reports the failure when listing fails partway", async () => {
		const source = addSource({ resourceTypes: ["contacts"] });
		stubGoHighLevel(account([contact("c2"), contact("c1")]));
		await sync(source);

		stubGoHighLevel((request) =>
			request.query.get("startAfterId") === null ? page("contacts", [contact("c2")], 2) : apiError(400, "Bad Request"),
		);
		const result = await sync(source);

		expect(result.failures[0]?.message).toContain("GoHighLevel contacts list failed");
		expect(result.failures[0]?.message).toContain("stale contacts were not purged");
		expect(liveIds(source.id, "source_gohighlevel_contact")).toEqual(["c1", "c2"]);
	});

	it("stops the whole sync when GoHighLevel rejects the token", async () => {
		const source = addSource();
		const requests = stubGoHighLevel(() => apiError(401, "Unauthorized"));

		const result = await sync(source);

		expect(requests).toHaveLength(1);
		expect(result.failures.map((failure) => failure.message)).toEqual([
			expect.stringContaining("GoHighLevel rejected the token"),
			expect.stringContaining("opportunities were not attempted"),
		]);
	});

	it("continues with other resources when the token cannot read one", async () => {
		const source = addSource();
		stubGoHighLevel((request) =>
			request.path === "/contacts/" ? apiError(403, "Forbidden") : paged("opportunities", [[opportunity]], 1)(request),
		);

		const result = await sync(source);

		expect(result.failures).toHaveLength(1);
		expect(result.failures[0]?.message).toContain("cannot read contacts");
		expect(liveIds(source.id, "source_gohighlevel_opportunity")).toEqual(["opp_1"]);
	});

	it("retries rate-limited requests and honors Retry-After", async () => {
		const source = addSource({ resourceTypes: ["contacts"] });
		let calls = 0;
		stubGoHighLevel((request) => {
			calls++;
			return calls === 1 ? apiError(429, "Too Many Requests", { "retry-after": "10" }) : account()(request);
		});

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(sleeps).toEqual([10_000]);
	});

	it("stops without recording failures when the sync is cancelled", async () => {
		const source = addSource();
		let live = true;
		const requests = stubGoHighLevel((request) => {
			live = false;
			return account()(request);
		});

		const result = await sync(source, () => live);

		expect(requests).toHaveLength(1);
		expect(result.failures).toEqual([]);
		expect(allRows(source.id).some((row) => row.source_kind === "source_gohighlevel_failure")).toBe(false);
	});
});

describe("gohighlevel-source-provider conversations", () => {
	let dir = "";
	let previousSignetPath: string | undefined;

	beforeEach(async () => {
		dir = mkdtempSync(join(tmpdir(), "signet-gohighlevel-conversations-"));
		previousSignetPath = process.env.SIGNET_PATH;
		process.env.SIGNET_PATH = dir;
		mkdirSync(join(dir, "memory"), { recursive: true });
		closeDbAccessor();
		initDbAccessor(join(dir, "memory", "memories.db"));
		setSecretKeyringAdapterForTests(memoryKeyring());
		await putSecret("GHL_TOKEN", TOKEN);
		setGoHighLevelSleepForTest(async () => {});
	});

	afterEach(async () => {
		globalThis.fetch = originalFetch;
		setGoHighLevelSleepForTest(null);
		setSecretKeyringAdapterForTests(null);
		await closeDbAccessor();
		if (previousSignetPath === undefined) Reflect.deleteProperty(process.env, "SIGNET_PATH");
		else process.env.SIGNET_PATH = previousSignetPath;
		rmSync(dir, { recursive: true, force: true });
	});

	function addSource(): SignetSourceEntry {
		const added = addGoHighLevelSource(
			{ tokenRef: "GHL_TOKEN", locationId: LOCATION, resourceTypes: ["conversations"] },
			dir,
		);
		if (added.ok === false) throw new Error(added.error);
		return added.source;
	}

	async function sync(source: SignetSourceEntry) {
		const result = await goHighLevelSourceProvider.sync?.({
			source,
			agentsDir: dir,
			agentId: "default",
			shouldContinue: () => true,
		});
		if (!result) throw new Error("GoHighLevel provider has no sync");
		return result;
	}

	function conversation(id: string, lastMessageDate: number, extra: Readonly<Record<string, unknown>> = {}) {
		return {
			id,
			contactId: `contact-${id}`,
			contactName: `Owner ${id}`,
			phone: "+15550100",
			type: "TYPE_PHONE",
			lastMessageType: "TYPE_SMS",
			lastMessageDirection: "inbound",
			lastMessageDate,
			dateUpdated: lastMessageDate,
			dateAdded: 1_767_225_600_000,
			sort: [lastMessageDate],
			...extra,
		};
	}

	const thread = [
		{
			id: "msg-3",
			messageType: "TYPE_EMAIL",
			direction: "outbound",
			contentType: "text/html",
			body: "Preview only",
			dateAdded: "2026-01-03T00:00:00.000Z",
			meta: { email: { messageIds: ["email-3"], subject: "Your renewal" } },
		},
		{
			id: "msg-2",
			messageType: "TYPE_SMS",
			direction: "inbound",
			status: "delivered",
			body: "Can I pause next month?",
			dateAdded: "2026-01-02T00:00:00.000Z",
			attachments: ["https://files.example/secret-attachment.png"],
		},
		{ id: "msg-1", messageType: "TYPE_ACTIVITY_OPPORTUNITY", body: "", dateAdded: "2026-01-01T00:00:00.000Z" },
	];

	function inbox(pages: readonly (readonly ReturnType<typeof conversation>[])[], total: number): Route {
		return (request) => {
			if (request.path === "/conversations/search") {
				const after = request.query.get("startAfterDate");
				const index =
					after === null ? 0 : pages.findIndex((items) => String(items[items.length - 1]?.sort[0]) === after) + 1;
				return Response.json({ conversations: pages[index] ?? [], total });
			}
			if (/^\/conversations\/[^/]+\/messages$/.test(request.path)) {
				return Response.json({ messages: { lastMessageId: "msg-1", nextPage: false, messages: thread } });
			}
			if (request.path === "/conversations/messages/email/email-3") {
				return Response.json({
					emailMessage: {
						id: "email-3",
						subject: "Your renewal",
						body: "<p>Hello&nbsp;there,</p><p>Your plan renews on the 1st.</p>",
					},
				});
			}
			return apiError(404, "Not Found");
		};
	}

	it("indexes each conversation with its SMS and full email text in chronological order", async () => {
		const source = addSource();
		const requests = stubGoHighLevel(inbox([[conversation("cv1", 1_767_400_000_000)]], 1));

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(result).toMatchObject({ indexed: 1, scanned: 1 });
		expect(requests[0]?.query.get("locationId")).toBe(LOCATION);
		expect(requests.map((request) => request.path)).toEqual([
			"/conversations/search",
			"/conversations/cv1/messages",
			"/conversations/messages/email/email-3",
			"/conversations/search",
		]);
		const [row] = liveRows(source.id);
		expect(row?.source_kind).toBe("source_gohighlevel_conversation");
		expect(row?.source_path).toBe(`gohighlevel://sources/${source.id}/conversations/cv1`);
		const content = row?.content ?? "";
		expect(content).toContain("# Conversation with Owner cv1");
		expect(content).toContain("- Contact: contact-cv1");
		expect(content).toContain("## Messages");
		expect(content).toContain("### 2026-01-02T00:00:00.000Z · inbound · sms · delivered");
		expect(content).toContain("Can I pause next month?");
		expect(content).toContain("Attachments: 1");
		expect(content).toContain("Subject: Your renewal");
		expect(content).toContain("Hello there,\nYour plan renews on the 1st.");
		expect(content).not.toContain("Preview only");
		expect(content).not.toContain("secret-attachment");
		expect(content.indexOf("Can I pause")).toBeLessThan(content.indexOf("Your plan renews"));
		expect(JSON.parse(row?.source_meta_json ?? "{}")).toMatchObject({
			resource: "conversations",
			contactId: "contact-cv1",
			versionKey: "1767400000000:1767400000000",
			messages: 2,
			olderMessagesOmitted: false,
			emailPreviewsOnly: 0,
		});
	});

	it("refetches messages only for conversations whose last activity changed", async () => {
		const source = addSource();
		stubGoHighLevel(inbox([[conversation("cv2", 1_767_500_000_000), conversation("cv1", 1_767_400_000_000)]], 2));
		await sync(source);

		const requests = stubGoHighLevel(
			inbox([[conversation("cv2", 1_767_600_000_000), conversation("cv1", 1_767_400_000_000)]], 2),
		);
		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(requests.filter((request) => request.path.endsWith("/messages")).map((request) => request.path)).toEqual([
			"/conversations/cv2/messages",
		]);
	});

	it("follows startAfterDate cursors and purges conversations gone after a complete enumeration", async () => {
		const source = addSource();
		const cv3 = conversation("cv3", 1_767_600_000_000);
		const cv2 = conversation("cv2", 1_767_500_000_000);
		const cv1 = conversation("cv1", 1_767_400_000_000);
		const requests = stubGoHighLevel(inbox([[cv3, cv2], [cv1]], 3));
		await sync(source);
		expect(
			requests
				.filter((request) => request.path === "/conversations/search")
				.map((request) => request.query.get("startAfterDate")),
		).toEqual([null, "1767500000000", "1767400000000"]);
		expect(liveRows(source.id).map((row) => row.source_external_id)).toEqual(["cv1", "cv2", "cv3"]);

		stubGoHighLevel(inbox([[cv3, cv1]], 2));
		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(liveRows(source.id).map((row) => row.source_external_id)).toEqual(["cv1", "cv3"]);
	});

	it("reports conversations whose messages could not be fetched and indexes the rest", async () => {
		const source = addSource();
		const inner = inbox([[conversation("cv2", 1_767_500_000_000), conversation("cv1", 1_767_400_000_000)]], 2);
		stubGoHighLevel((request) =>
			request.path === "/conversations/cv2/messages" ? apiError(503, "Service Unavailable") : inner(request),
		);

		const result = await sync(source);

		expect(result.failures.map((failure) => failure.message)).toEqual([
			expect.stringContaining("1 GoHighLevel conversation(s) could not be fetched"),
		]);
		expect(
			liveRows(source.id)
				.filter((row) => row.source_kind === "source_gohighlevel_conversation")
				.map((row) => row.source_external_id),
		).toEqual(["cv1"]);
	});
});

describe("gohighlevel-source-fetch", () => {
	it("rejects objects without an id and falls back through contact names", () => {
		expect(parseGoHighLevelRecord("contacts", { contactName: "No id" })).toBeNull();
		expect(parseGoHighLevelRecord("contacts", { id: "c9", email: "c9@example.com" })?.title).toBe("c9@example.com");
		expect(parseGoHighLevelRecord("opportunities", { id: "o9" })?.title).toBe("Opportunity o9");
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
