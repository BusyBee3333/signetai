import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type SignetSourceEntry, addStripeSource } from "@signet/core";
import type { SecretKeyringAdapter } from "@signet/core";
import { closeDbAccessor, getDbAccessor, initDbAccessor } from "./db-accessor";
import { putSecret, setSecretKeyringAdapterForTests } from "./secrets";
import { STRIPE_API_VERSION, money, parseStripeRecord, setStripeSleepForTest } from "./stripe-source-fetch";
import { setStripeSyncDeadlineForTest, stripeSourceProvider } from "./stripe-source-provider";

const originalFetch = globalThis.fetch;
const TOKEN = "stripe-test-key-value";
const CARD_LAST4 = "4242";

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
		account: "stripe-source-test",
		get: read,
		getStatus: read,
		async set(value: string) {
			stored = value;
			return { state: "found" as const, value };
		},
	};
}

function stubStripe(route: Route): RecordedRequest[] {
	const requests: RecordedRequest[] = [];
	globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
		const url = new URL(String(input));
		const request: RecordedRequest = {
			path: url.pathname.replace(/^\/v1/, ""),
			query: url.searchParams,
			headers: new Headers(init?.headers),
		};
		requests.push(request);
		return await route(request);
	}) as typeof fetch;
	return requests;
}

function list(data: readonly unknown[], hasMore = false): Response {
	return Response.json({ object: "list", url: "/v1/x", has_more: hasMore, data });
}

function stripeError(status: number, code: string, headers: Record<string, string> = {}): Response {
	return Response.json(
		{ error: { type: "invalid_request_error", code, message: `${code} message` } },
		{ status, headers },
	);
}

function customer(id: string, extra: Readonly<Record<string, unknown>> = {}): Record<string, unknown> {
	return {
		id,
		object: "customer",
		created: 1_767_225_600,
		livemode: true,
		name: `Gym ${id}`,
		email: `${id}@example.com`,
		currency: "usd",
		delinquent: false,
		balance: 0,
		metadata: {},
		...extra,
	};
}

const subscription = {
	id: "sub_1",
	object: "subscription",
	created: 1_767_225_600,
	livemode: true,
	customer: "cus_1",
	status: "active",
	cancel_at_period_end: false,
	collection_method: "charge_automatically",
	latest_invoice: { id: "in_1", object: "invoice" },
	items: {
		object: "list",
		data: [
			{
				id: "si_1",
				quantity: 1,
				current_period_start: 1_767_225_600,
				current_period_end: 1_769_904_000,
				price: {
					id: "price_1",
					currency: "usd",
					unit_amount: 29_700,
					product: "prod_1",
					recurring: { interval: "month", interval_count: 1 },
				},
			},
		],
	},
	metadata: { plan: "pro" },
};

const invoice = {
	id: "in_1",
	object: "invoice",
	created: 1_767_225_600,
	livemode: true,
	customer: "cus_1",
	customer_name: "Gym cus_1",
	customer_email: "cus_1@example.com",
	number: "BLG-0001",
	status: "open",
	currency: "usd",
	total: 29_700,
	amount_due: 29_700,
	amount_paid: 0,
	amount_remaining: 29_700,
	attempt_count: 2,
	hosted_invoice_url: "https://invoice.stripe.com/i/secret-link",
	parent: { type: "subscription_details", subscription_details: { subscription: "sub_1" } },
};

const charge = {
	id: "ch_1",
	object: "charge",
	created: 1_767_225_600,
	livemode: true,
	customer: "cus_1",
	amount: 29_700,
	amount_refunded: 0,
	currency: "usd",
	status: "failed",
	paid: false,
	refunded: false,
	disputed: false,
	failure_code: "card_declined",
	failure_message: "Your card was declined.",
	payment_intent: "pi_1",
	receipt_url: "https://pay.stripe.com/receipts/secret-link",
	payment_method_details: {
		type: "card",
		card: { brand: "visa", last4: CARD_LAST4, exp_month: 12, exp_year: 2030, fingerprint: "fp_secret" },
	},
};

function fullAccount(customers: readonly unknown[] = [customer("cus_1")]): Route {
	return (request) => {
		if (request.path === "/customers") return list(customers);
		if (request.path === "/subscriptions") return list([subscription]);
		if (request.path === "/invoices") return list([invoice]);
		if (request.path === "/charges") return list([charge]);
		return stripeError(404, "resource_missing");
	};
}

describe("stripe-source-provider", () => {
	let dir = "";
	let previousSignetPath: string | undefined;
	let sleeps: number[] = [];

	beforeEach(async () => {
		dir = mkdtempSync(join(tmpdir(), "signet-stripe-source-"));
		previousSignetPath = process.env.SIGNET_PATH;
		process.env.SIGNET_PATH = dir;
		mkdirSync(join(dir, "memory"), { recursive: true });
		closeDbAccessor();
		initDbAccessor(join(dir, "memory", "memories.db"));
		setSecretKeyringAdapterForTests(memoryKeyring());
		await putSecret("STRIPE_KEY", TOKEN);
		sleeps = [];
		setStripeSleepForTest(async (ms) => {
			sleeps.push(ms);
		});
	});

	afterEach(async () => {
		globalThis.fetch = originalFetch;
		setStripeSleepForTest(null);
		setStripeSyncDeadlineForTest(null);
		setSecretKeyringAdapterForTests(null);
		await closeDbAccessor();
		if (previousSignetPath === undefined) Reflect.deleteProperty(process.env, "SIGNET_PATH");
		else process.env.SIGNET_PATH = previousSignetPath;
		rmSync(dir, { recursive: true, force: true });
	});

	function addSource(input: { resourceTypes?: readonly string[]; maxItemsPerType?: number } = {}): SignetSourceEntry {
		const added = addStripeSource({ tokenRef: "STRIPE_KEY", ...input }, dir);
		if (added.ok === false) throw new Error(added.error);
		return added.source;
	}

	async function sync(source: SignetSourceEntry, shouldContinue: () => boolean = () => true) {
		const result = await stripeSourceProvider.sync?.({ source, agentsDir: dir, agentId: "default", shouldContinue });
		if (!result) throw new Error("Stripe provider has no sync");
		return result;
	}

	it("indexes customers, subscriptions, invoices, and charges as attributable source artifacts", async () => {
		const source = addSource();
		const requests = stubStripe(fullAccount());

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(result).toMatchObject({ indexed: 4, scanned: 4, total: 4 });
		expect(requests.map((request) => request.path)).toEqual(["/customers", "/subscriptions", "/invoices", "/charges"]);
		for (const request of requests) {
			expect(request.headers.get("authorization")).toBe(`Bearer ${TOKEN}`);
			expect(request.headers.get("stripe-version")).toBe(STRIPE_API_VERSION);
			expect(request.query.get("limit")).toBe("100");
		}
		expect(requests[1]?.query.get("status")).toBe("all");

		const rows = liveRows(source.id);
		expect(rows.map((row) => [row.source_kind, row.source_path])).toEqual([
			["source_stripe_charge", `stripe://sources/${source.id}/charges/ch_1`],
			["source_stripe_customer", `stripe://sources/${source.id}/customers/cus_1`],
			["source_stripe_invoice", `stripe://sources/${source.id}/invoices/in_1`],
			["source_stripe_subscription", `stripe://sources/${source.id}/subscriptions/sub_1`],
		]);
		const byId = new Map(rows.map((row) => [row.source_external_id, row]));
		expect(byId.get("cus_1")?.content).toContain("# Gym cus_1");
		expect(byId.get("cus_1")?.content).toContain("- Email: cus_1@example.com");
		expect(byId.get("sub_1")?.content).toContain("- Status: active");
		expect(byId.get("sub_1")?.content).toContain("- Item 1: price_1, product prod_1, 297.00 USD per month, quantity 1");
		expect(byId.get("sub_1")?.content).toContain("- Current period end: 2026-02-01T00:00:00.000Z");
		expect(byId.get("sub_1")?.content).toContain("- Latest invoice: in_1");
		expect(byId.get("in_1")?.content).toContain("# Invoice BLG-0001");
		expect(byId.get("in_1")?.content).toContain("- Amount remaining: 297.00 USD");
		expect(byId.get("in_1")?.content).toContain("- Subscription: sub_1");
		expect(byId.get("ch_1")?.content).toContain("- Failure code: card_declined");
		expect(byId.get("ch_1")?.content).toContain("- Payment method type: card");
		expect(byId.get("in_1")?.source_parent_path).toBe(`stripe://sources/${source.id}/invoices`);
		expect(JSON.parse(byId.get("ch_1")?.source_meta_json ?? "{}")).toMatchObject({
			provider: "stripe",
			resource: "charges",
			objectId: "ch_1",
			customerId: "cus_1",
			livemode: true,
		});
	});

	it("never stores the API key, card details, or bearer payment links", async () => {
		const source = addSource();
		stubStripe(fullAccount());

		await sync(source);

		const stored = allRows(source.id)
			.map((row) => `${row.content}\n${row.source_meta_json ?? ""}`)
			.join("\n");
		for (const secret of [TOKEN, CARD_LAST4, "visa", "fp_secret", "secret-link", "exp_year"]) {
			expect(stored).not.toContain(secret);
		}
	});

	it("follows starting_after cursors until Stripe reports the end of the list", async () => {
		const source = addSource({ resourceTypes: ["customers"] });
		const requests = stubStripe((request) => {
			const after = request.query.get("starting_after");
			if (after === null) return list([customer("cus_3"), customer("cus_2")], true);
			if (after === "cus_2") return list([customer("cus_1")], false);
			return stripeError(400, "unexpected_cursor");
		});

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(result).toMatchObject({ indexed: 3, scanned: 3 });
		expect(requests.map((request) => request.query.get("starting_after"))).toEqual([null, "cus_2"]);
		expect(liveRows(source.id).map((row) => row.source_external_id)).toEqual(["cus_1", "cus_2", "cus_3"]);
	});

	it("leaves unchanged objects untouched and rewrites changed ones on the next sync", async () => {
		const source = addSource({ resourceTypes: ["customers"] });
		stubStripe(() => list([customer("cus_1"), customer("cus_2")]));
		await sync(source);
		const before = new Map(allRows(source.id).map((row) => [row.source_external_id, row.updated_at]));
		await new Promise((resolve) => setTimeout(resolve, 5));

		stubStripe(() => list([customer("cus_1"), customer("cus_2", { delinquent: true })]));
		const result = await sync(source);

		expect(result).toMatchObject({ indexed: 2, scanned: 2 });
		const after = new Map(allRows(source.id).map((row) => [row.source_external_id, row]));
		expect(after.get("cus_1")?.updated_at).toBe(before.get("cus_1"));
		expect(after.get("cus_2")?.updated_at).not.toBe(before.get("cus_2"));
		expect(after.get("cus_2")?.content).toContain("- Delinquent: yes");
	});

	it("purges objects that are gone after a complete enumeration", async () => {
		const source = addSource({ resourceTypes: ["customers"] });
		stubStripe(() => list([customer("cus_1"), customer("cus_2")]));
		await sync(source);

		stubStripe(() => list([customer("cus_1")]));
		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(liveRows(source.id).map((row) => row.source_external_id)).toEqual(["cus_1"]);
		expect(allRows(source.id).find((row) => row.source_external_id === "cus_2")?.is_deleted).toBe(1);
	});

	it("keeps existing objects when the configured cap ends enumeration early", async () => {
		const source = addSource({ resourceTypes: ["customers"] });
		stubStripe(() => list([customer("cus_2"), customer("cus_1")]));
		await sync(source);

		const capped = addSource({ resourceTypes: ["customers"], maxItemsPerType: 1 });
		const requests = stubStripe(() => list([customer("cus_2")], true));
		const result = await sync(capped);

		expect(requests[0]?.query.get("limit")).toBe("1");
		expect(result).toMatchObject({ scanned: 1 });
		expect(liveRows(source.id).map((row) => row.source_external_id)).toEqual(["cus_1", "cus_2"]);
	});

	it("keeps existing objects and reports the failure when listing fails partway", async () => {
		const source = addSource({ resourceTypes: ["customers"] });
		stubStripe(() => list([customer("cus_2"), customer("cus_1")]));
		await sync(source);

		stubStripe((request) =>
			request.query.get("starting_after") === null ? list([customer("cus_2")], true) : stripeError(400, "bad_request"),
		);
		const result = await sync(source);

		expect(result.failures.map((failure) => failure.message)).toEqual([
			expect.stringContaining("Stripe customers list failed"),
		]);
		expect(result.failures[0]?.message).toContain("stale customers were not purged");
		expect(
			liveRows(source.id)
				.filter((row) => row.source_kind === "source_stripe_customer")
				.map((row) => row.source_external_id),
		).toEqual(["cus_1", "cus_2"]);
		expect(allRows(source.id).some((row) => row.source_kind === "source_stripe_failure" && row.is_deleted === 0)).toBe(
			true,
		);
	});

	it("stops the whole sync when Stripe rejects the API key", async () => {
		const source = addSource();
		const requests = stubStripe(() => stripeError(401, "invalid_api_key"));

		const result = await sync(source);

		expect(requests.map((request) => request.path)).toEqual(["/customers"]);
		expect(result.scanned).toBe(0);
		expect(result.failures.map((failure) => failure.message)).toEqual([
			expect.stringContaining("Stripe rejected the API key"),
			expect.stringContaining("subscriptions, invoices, charges were not attempted"),
		]);
		expect(liveRows(source.id).every((row) => row.source_kind === "source_stripe_failure")).toBe(true);
		expect(allRows(source.id).every((row) => !row.content.includes(TOKEN))).toBe(true);
	});

	it("continues with other resources when a restricted key cannot read one", async () => {
		const source = addSource({ resourceTypes: ["customers", "charges"] });
		stubStripe((request) => (request.path === "/customers" ? stripeError(403, "permission_denied") : list([charge])));

		const result = await sync(source);

		expect(result.scanned).toBe(1);
		expect(result.failures).toHaveLength(1);
		expect(result.failures[0]?.message).toContain("cannot read customers");
		expect(liveRows(source.id).some((row) => row.source_external_id === "ch_1")).toBe(true);
	});

	it("retries rate-limited requests and honors Retry-After", async () => {
		const source = addSource({ resourceTypes: ["customers"] });
		let calls = 0;
		stubStripe(() => {
			calls++;
			return calls === 1 ? stripeError(429, "rate_limit", { "retry-after": "7" }) : list([customer("cus_1")]);
		});

		const result = await sync(source);

		expect(result.failures).toEqual([]);
		expect(calls).toBe(2);
		expect(sleeps).toEqual([7_000]);
	});

	it("reports objects with an unexpected shape and does not purge on that evidence", async () => {
		const source = addSource({ resourceTypes: ["customers"] });
		stubStripe(() => list([customer("cus_1"), customer("cus_2")]));
		await sync(source);

		stubStripe(() => list([customer("cus_1"), { id: "cus_2", object: "something_else" }]));
		const result = await sync(source);

		expect(result.failures[0]?.message).toContain("Skipped 1 Stripe customers object(s)");
		expect(liveRows(source.id).filter((row) => row.source_kind === "source_stripe_customer")).toHaveLength(2);
	});

	it("stops without recording failures when the sync is cancelled", async () => {
		const source = addSource();
		let live = true;
		const requests = stubStripe((request) => {
			if (request.path === "/customers") {
				live = false;
				return list([customer("cus_1")], true);
			}
			return list([]);
		});

		const result = await sync(source, () => live);

		expect(requests.map((request) => request.path)).toEqual(["/customers"]);
		expect(result.failures).toEqual([]);
		expect(allRows(source.id).some((row) => row.source_kind === "source_stripe_failure")).toBe(false);
	});

	it("stops at the sync deadline and reports what was not attempted", async () => {
		const source = addSource({ resourceTypes: ["customers", "charges"] });
		setStripeSyncDeadlineForTest(-1);
		const requests = stubStripe(fullAccount());

		const result = await sync(source);

		expect(requests).toEqual([]);
		expect(result.failures.map((failure) => failure.message)).toEqual([
			expect.stringContaining("deadline"),
			expect.stringContaining("charges were not attempted"),
		]);
	});
});

describe("stripe-source-fetch", () => {
	it("rejects objects of the wrong type and deleted objects", () => {
		expect(parseStripeRecord("customers", { id: "cus_1", object: "charge" })).toBeNull();
		expect(parseStripeRecord("customers", { id: "cus_1", object: "customer", deleted: true })).toBeNull();
		expect(parseStripeRecord("customers", { object: "customer" })).toBeNull();
	});

	it("reads subscription periods and invoice subscriptions from either API shape", () => {
		const legacy = parseStripeRecord("subscriptions", {
			id: "sub_old",
			object: "subscription",
			customer: { id: "cus_9" },
			current_period_end: 1_769_904_000,
		});
		expect(legacy?.customerId).toBe("cus_9");
		expect(legacy?.fields).toContainEqual({ name: "Current period end", value: "2026-02-01T00:00:00.000Z" });
		const legacyInvoice = parseStripeRecord("invoices", { id: "in_old", object: "invoice", subscription: "sub_old" });
		expect(legacyInvoice?.fields).toContainEqual({ name: "Subscription", value: "sub_old" });
	});

	it("formats amounts by currency exponent", () => {
		expect(money(29_700, "usd")).toBe("297.00 USD");
		expect(money(5_000, "jpy")).toBe("5000 JPY");
		expect(money(1_234, "kwd")).toBe("1.234 KWD");
		expect(money(null, "usd")).toBe("");
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
