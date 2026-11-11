import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { closeDbAccessor, initDbAccessor } from "./db-accessor";
import { indexExternalMemoryArtifact } from "./memory-lineage";
import { extractIdentityKeys, lookupIdentity, normalizeEmail, normalizePhone } from "./source-identity";

const STRIPE = "stripe://sources/stripe:a";
const GHL = "gohighlevel://sources/gohighlevel:b";

async function put(
	agentId: string,
	path: string,
	kind: string,
	title: string,
	fields: Readonly<Record<string, string>>,
) {
	const harness = kind.split("_")[1] ?? "test";
	await indexExternalMemoryArtifact({
		agentId,
		harness,
		sourceId: `${harness}:x`,
		sourcePath: path,
		sourceKind: kind,
		sourceMtimeMs: 1_767_225_600_000,
		content: [`# ${title}`, "", ...Object.entries(fields).map(([name, value]) => `- ${name}: ${value}`)].join("\n"),
	});
}

describe("source-identity", () => {
	let dir = "";
	let previousSignetPath: string | undefined;

	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "signet-source-identity-"));
		previousSignetPath = process.env.SIGNET_PATH;
		process.env.SIGNET_PATH = dir;
		mkdirSync(join(dir, "memory"), { recursive: true });
		closeDbAccessor();
		initDbAccessor(join(dir, "memory", "memories.db"));
	});

	afterEach(async () => {
		await closeDbAccessor();
		if (previousSignetPath === undefined) Reflect.deleteProperty(process.env, "SIGNET_PATH");
		else process.env.SIGNET_PATH = previousSignetPath;
		rmSync(dir, { recursive: true, force: true });
	});

	async function seed(agentId = "default") {
		// One person: Stripe and GoHighLevel share only a phone; a second GoHighLevel contact shares only an email with the first.
		await put(agentId, `${STRIPE}/customers/cus_AAAA1111`, "source_stripe_customer", "Dana Whitfield", {
			Email: "Dana.Whitfield@Ironworks.example",
			Phone: "(555) 010-2233",
		});
		await put(
			agentId,
			`${STRIPE}/subscriptions/sub_BBBB2222`,
			"source_stripe_subscription",
			"Subscription sub_BBBB2222",
			{
				Customer: "cus_AAAA1111",
				Status: "past_due",
			},
		);
		await put(agentId, `${GHL}/contacts/ghlContact000001`, "source_gohighlevel_contact", "Dana W.", {
			Email: "dana@personal.example",
			Phone: "+15550102233",
		});
		await put(agentId, `${GHL}/contacts/ghlContact000002`, "source_gohighlevel_contact", "D. Whitfield", {
			Email: "dana@personal.example",
		});
		await put(
			agentId,
			`${GHL}/conversations/conv0000000001`,
			"source_gohighlevel_conversation",
			"Conversation with Dana W.",
			{
				Contact: "ghlContact000001",
				"Last message": "see you Tuesday",
			},
		);
		// Different people who must not be merged.
		await put(agentId, `${GHL}/contacts/ghlContact000003`, "source_gohighlevel_contact", "Dana Whitfield", {
			Email: "other.dana@elsewhere.example",
			Phone: "+15550109999",
		});
		await put(agentId, `${STRIPE}/customers/cus_CCCC3333`, "source_stripe_customer", "Sam Ortiz", {
			Email: "dana.whitfield@ironworks.example.org",
			Phone: "+1 555 010 2234",
		});
		await put(agentId, `${STRIPE}/charges/ch_DDDD4444`, "source_stripe_charge", "Charge ch_DDDD4444", {
			Customer: "cus_AAAA11119",
			Description: "mentions dana.whitfield@ironworks.example in free text only",
		});
	}

	it("normalizes emails and phone numbers into comparable keys", () => {
		expect(normalizeEmail(" Dana@Example.COM ")).toBe("dana@example.com");
		expect(normalizeEmail("not an email")).toBeNull();
		expect(normalizePhone("(555) 010-2233")).toBe("5550102233");
		expect(normalizePhone("+1 555.010.2233")).toBe("5550102233");
		expect(normalizePhone("010-2233")).toBeNull();
		expect(normalizePhone("order 5550102233 shipped")).toBeNull();
		expect(
			extractIdentityKeys(
				"# T\n\n- Email: A@b.co\n- Additional emails: c@d.co, e@f.co\n- Phone: 555-010-2233\n- Notes: z@y.co",
			),
		).toEqual([
			{ type: "email", value: "a@b.co" },
			{ type: "email", value: "c@d.co" },
			{ type: "email", value: "e@f.co" },
			{ type: "phone", value: "5550102233" },
		]);
	});

	it("joins one person's records across tools through shared phone, shared email, and tool IDs", async () => {
		await seed();
		const result = await lookupIdentity({ agentId: "default", query: "dana.whitfield@ironworks.example" });
		const byPath = new Map(result.records.map((record) => [record.sourcePath, record]));

		expect([...byPath.keys()].sort()).toEqual(
			[
				`${STRIPE}/customers/cus_AAAA1111`,
				`${STRIPE}/subscriptions/sub_BBBB2222`,
				`${GHL}/contacts/ghlContact000001`,
				`${GHL}/contacts/ghlContact000002`,
				`${GHL}/conversations/conv0000000001`,
			].sort(),
		);
		expect(byPath.get(`${STRIPE}/customers/cus_AAAA1111`)?.match).toBe("email");
		expect(byPath.get(`${GHL}/contacts/ghlContact000001`)).toMatchObject({
			match: "phone",
			matchedValue: "5550102233",
			provider: "gohighlevel",
		});
		expect(byPath.get(`${GHL}/contacts/ghlContact000002`)).toMatchObject({
			match: "email",
			matchedValue: "dana@personal.example",
		});
		expect(byPath.get(`${STRIPE}/subscriptions/sub_BBBB2222`)).toMatchObject({
			match: "id",
			matchedValue: "cus_AAAA1111",
		});
		expect(byPath.get(`${GHL}/conversations/conv0000000001`)).toMatchObject({
			match: "id",
			matchedValue: "ghlContact000001",
		});
		expect(result.truncated).toBe(false);
	});

	it("reports a same-name record with no shared key as possible, never as confirmed", async () => {
		await seed();
		const result = await lookupIdentity({ agentId: "default", query: "Dana Whitfield" });

		// Both records are named Dana Whitfield, so both seed the lookup and neither is silently preferred.
		expect(
			result.records
				.filter((record) => record.match === "query")
				.map((record) => record.sourcePath)
				.sort(),
		).toEqual([`${GHL}/contacts/ghlContact000003`, `${STRIPE}/customers/cus_AAAA1111`].sort());

		const byEmail = await lookupIdentity({ agentId: "default", query: "dana@personal.example" });
		expect(byEmail.records.map((record) => record.sourcePath)).not.toContain(`${GHL}/contacts/ghlContact000003`);
		expect(byEmail.possible).toEqual([
			{
				sourcePath: `${GHL}/contacts/ghlContact000003`,
				sourceKind: "source_gohighlevel_contact",
				sourceId: "gohighlevel:x",
				provider: "gohighlevel",
				title: "Dana Whitfield",
				reason: "Same name, but no shared email address or phone number",
			},
		]);
	});

	it("does not confirm near-miss emails, near-miss phones, longer IDs, or free-text mentions", async () => {
		await seed();
		const result = await lookupIdentity({ agentId: "default", query: "+1 (555) 010-2233" });
		const paths = result.records.map((record) => record.sourcePath);
		expect(paths).toContain(`${STRIPE}/customers/cus_AAAA1111`);
		expect(paths).not.toContain(`${STRIPE}/customers/cus_CCCC3333`);
		expect(paths).not.toContain(`${STRIPE}/charges/ch_DDDD4444`);
	});

	it("looks a person up by a tool ID", async () => {
		await seed();
		const result = await lookupIdentity({ agentId: "default", query: "cus_AAAA1111" });
		expect(result.records.map((record) => record.sourcePath)).toContain(`${GHL}/contacts/ghlContact000001`);
	});

	it("stays inside the requesting agent's scope", async () => {
		await seed("other-agent");
		const result = await lookupIdentity({ agentId: "default", query: "dana.whitfield@ironworks.example" });
		expect(result.records).toEqual([]);
		expect(result.possible).toEqual([]);
	});
});
