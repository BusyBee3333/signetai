import type { StripeSourceResourceType } from "@signet/core";

export const STRIPE_API_BASE = "https://api.stripe.com/v1";
export const STRIPE_API_VERSION = "2026-09-30.endive";
export const STRIPE_MAX_RESPONSE_BYTES = 16 * 1024 * 1024;
export const STRIPE_PAGE_SIZE = 100;
const REQUEST_TIMEOUT_MS = 30_000;
const MAX_ATTEMPTS = 4;
const RETRY_BASE_DELAY_MS = 1_000;
const MAX_RETRY_DELAY_MS = 60_000;
const MAX_FIELD_VALUE_CHARS = 2_000;
const MAX_METADATA_ENTRIES = 25;
const MAX_LINE_ITEMS = 20;
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);
const ZERO_DECIMAL_CURRENCIES = new Set([
	"bif",
	"clp",
	"djf",
	"gnf",
	"jpy",
	"kmf",
	"krw",
	"mga",
	"pyg",
	"rwf",
	"ugx",
	"vnd",
	"vuv",
	"xaf",
	"xof",
	"xpf",
]);
const THREE_DECIMAL_CURRENCIES = new Set(["bhd", "jod", "kwd", "omr", "tnd"]);

const CANCEL_POLL_MS = 250;

type Sleep = (ms: number, shouldContinue: () => boolean) => Promise<void>;
type ShouldContinue = () => boolean;

let stripeSleep: Sleep = cancellableSleep;

export function setStripeSleepForTest(sleep: Sleep | null): void {
	stripeSleep = sleep ?? cancellableSleep;
}

async function cancellableSleep(ms: number, shouldContinue: ShouldContinue): Promise<void> {
	const deadline = Date.now() + ms;
	while (Date.now() < deadline && shouldContinue()) {
		await new Promise((resolve) => setTimeout(resolve, Math.min(CANCEL_POLL_MS, deadline - Date.now())));
	}
}

export class StripeRequestError extends Error {
	readonly status: number;
	readonly code: string;
	readonly retryable: boolean;
	readonly retryAfterMs: number | null;

	constructor(
		message: string,
		options: {
			readonly status: number;
			readonly code: string;
			readonly retryable: boolean;
			readonly retryAfterMs?: number;
		},
	) {
		super(message);
		this.name = "StripeRequestError";
		this.status = options.status;
		this.code = options.code;
		this.retryable = options.retryable;
		this.retryAfterMs = options.retryAfterMs ?? null;
	}
}

export interface StripeRecordField {
	readonly name: string;
	readonly value: string;
}

export interface StripeRecord {
	readonly resource: StripeSourceResourceType;
	readonly id: string;
	readonly title: string;
	readonly createdMs: number | null;
	readonly customerId: string | null;
	readonly livemode: boolean;
	readonly fields: readonly StripeRecordField[];
}

export interface StripeListPage {
	readonly records: readonly StripeRecord[];
	readonly unparsed: number;
	readonly lastId: string | null;
	readonly hasMore: boolean;
}

const LIST_PATHS: Readonly<Record<StripeSourceResourceType, string>> = {
	customers: "/customers",
	subscriptions: "/subscriptions",
	invoices: "/invoices",
	charges: "/charges",
	refunds: "/refunds",
	disputes: "/disputes",
	credit_notes: "/credit_notes",
	coupons: "/coupons",
	products: "/products",
	prices: "/prices",
	payouts: "/payouts",
	balance_transactions: "/balance_transactions",
	events: "/events",
};

const OBJECT_NAMES: Readonly<Record<StripeSourceResourceType, string>> = {
	customers: "customer",
	subscriptions: "subscription",
	invoices: "invoice",
	charges: "charge",
	refunds: "refund",
	disputes: "dispute",
	credit_notes: "credit_note",
	coupons: "coupon",
	products: "product",
	prices: "price",
	payouts: "payout",
	balance_transactions: "balance_transaction",
	events: "event",
};

export async function listStripePage(
	token: string,
	resource: StripeSourceResourceType,
	startingAfter: string | null,
	limit: number,
	shouldContinue: ShouldContinue,
): Promise<StripeListPage> {
	const query = new URLSearchParams({ limit: String(Math.max(1, Math.min(STRIPE_PAGE_SIZE, limit))) });
	if (resource === "subscriptions") query.set("status", "all");
	// Refunds and disputes name only their charge; expanding it yields the customer they belong to.
	if (resource === "refunds" || resource === "disputes") query.append("expand[]", "data.charge");
	if (startingAfter) query.set("starting_after", startingAfter);
	const body = asRecord(await stripeRequest(token, `${LIST_PATHS[resource]}?${query.toString()}`, shouldContinue));
	if (!Array.isArray(body.data)) {
		throw new StripeRequestError("Stripe list response has no data array", {
			status: 200,
			code: "invalid_response",
			retryable: false,
		});
	}
	const records: StripeRecord[] = [];
	let unparsed = 0;
	let lastId: string | null = null;
	for (const item of body.data) {
		const id = asRecord(item).id;
		if (typeof id === "string" && id) lastId = id;
		const record = parseStripeRecord(resource, item);
		if (record) records.push(record);
		else unparsed++;
	}
	return { records, unparsed, lastId, hasMore: body.has_more === true };
}

async function stripeRequest(token: string, pathAndQuery: string, shouldContinue: ShouldContinue): Promise<unknown> {
	let lastError: StripeRequestError | null = null;
	for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
		if (lastError) await stripeSleep(retryDelay(lastError, attempt), shouldContinue);
		if (!shouldContinue()) throw cancelledRequest();
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
		const cancelPoll = setInterval(() => {
			if (!shouldContinue()) controller.abort();
		}, CANCEL_POLL_MS);
		try {
			const response = await fetch(`${STRIPE_API_BASE}${pathAndQuery}`, {
				method: "GET",
				headers: {
					Authorization: `Bearer ${token}`,
					"Stripe-Version": STRIPE_API_VERSION,
					Accept: "application/json",
				},
				signal: controller.signal,
			});
			const text = await readBoundedText(response);
			if (response.ok) return parseJson(text, response.status);
			const error = responseError(response, parseErrorBody(text));
			if (!error.retryable) throw error;
			lastError = error;
		} catch (err) {
			if (err instanceof StripeRequestError && !err.retryable) throw err;
			if (!shouldContinue()) throw cancelledRequest();
			lastError =
				err instanceof StripeRequestError
					? err
					: new StripeRequestError(
							controller.signal.aborted
								? `Stripe request timed out after ${REQUEST_TIMEOUT_MS}ms`
								: `Stripe request failed: ${err instanceof Error ? err.message : String(err)}`,
							{ status: 0, code: controller.signal.aborted ? "timeout" : "network", retryable: true },
						);
		} finally {
			clearTimeout(timeout);
			clearInterval(cancelPoll);
		}
	}
	throw lastError ?? new StripeRequestError("Stripe request failed", { status: 0, code: "network", retryable: true });
}

function responseError(response: Response, body: unknown): StripeRequestError {
	const error = asRecord(asRecord(body).error);
	const code =
		typeof error.code === "string"
			? error.code
			: typeof error.type === "string"
				? error.type
				: `http_${response.status}`;
	const detail = typeof error.message === "string" ? error.message : response.statusText;
	const retryAfter = Number(response.headers.get("retry-after") ?? Number.NaN);
	return new StripeRequestError(`Stripe API ${response.status} ${code}: ${detail}`.trim(), {
		status: response.status,
		code,
		retryable: RETRYABLE_STATUSES.has(response.status),
		...(Number.isFinite(retryAfter) && retryAfter >= 0 ? { retryAfterMs: retryAfter * 1_000 } : {}),
	});
}

function retryDelay(error: StripeRequestError, attempt: number): number {
	const backoff = RETRY_BASE_DELAY_MS * 2 ** (attempt - 1);
	return Math.min(MAX_RETRY_DELAY_MS, Math.max(error.retryAfterMs ?? 0, backoff));
}

function cancelledRequest(): StripeRequestError {
	return new StripeRequestError("Stripe request cancelled", { status: 0, code: "cancelled", retryable: false });
}

async function readBoundedText(response: Response): Promise<string> {
	const declared = Number(response.headers.get("content-length"));
	if (Number.isFinite(declared) && declared > STRIPE_MAX_RESPONSE_BYTES) {
		await response.body?.cancel();
		throw oversizedResponse(response.status);
	}
	const reader = response.body?.getReader();
	if (!reader) return "";
	const chunks: Uint8Array[] = [];
	let total = 0;
	for (;;) {
		const next = await reader.read();
		if (next.done) break;
		total += next.value.byteLength;
		if (total > STRIPE_MAX_RESPONSE_BYTES) {
			await reader.cancel();
			throw oversizedResponse(response.status);
		}
		chunks.push(next.value);
	}
	const bytes = new Uint8Array(total);
	let offset = 0;
	for (const chunk of chunks) {
		bytes.set(chunk, offset);
		offset += chunk.byteLength;
	}
	return new TextDecoder().decode(bytes);
}

function parseJson(text: string, status: number): unknown {
	try {
		return JSON.parse(text);
	} catch {
		throw new StripeRequestError(`Stripe API ${status} returned invalid JSON`, {
			status,
			code: "invalid_response",
			retryable: false,
		});
	}
}

function parseErrorBody(text: string): unknown {
	try {
		return JSON.parse(text);
	} catch {
		return null;
	}
}

function oversizedResponse(status: number): StripeRequestError {
	return new StripeRequestError(`Stripe API response exceeded ${STRIPE_MAX_RESPONSE_BYTES} bytes`, {
		status,
		code: "response_too_large",
		retryable: false,
	});
}

export function parseStripeRecord(resource: StripeSourceResourceType, value: unknown): StripeRecord | null {
	const record = asRecord(value);
	const id = typeof record.id === "string" ? record.id : "";
	if (!id || record.object !== OBJECT_NAMES[resource] || record.deleted === true) return null;
	const created = typeof record.created === "number" && Number.isFinite(record.created) ? record.created * 1_000 : null;
	const base = { resource, id, createdMs: created, livemode: record.livemode === true };
	if (resource === "customers") {
		const name = text(record.name);
		const email = text(record.email);
		return { ...base, title: name || email || id, customerId: id, fields: customerFields(record) };
	}
	const customerId = expandableId(record.customer);
	if (resource === "subscriptions") {
		return { ...base, title: `Subscription ${id}`, customerId, fields: subscriptionFields(record, customerId) };
	}
	if (resource === "invoices") {
		const number = text(record.number);
		return { ...base, title: `Invoice ${number || id}`, customerId, fields: invoiceFields(record, customerId) };
	}
	if (resource === "charges") {
		return { ...base, title: `Charge ${id}`, customerId, fields: chargeFields(record, customerId) };
	}
	if (resource === "refunds" || resource === "disputes") {
		const owner = expandableId(asRecord(record.charge).customer);
		const label = resource === "refunds" ? "Refund" : "Dispute";
		return { ...base, title: `${label} ${id}`, customerId: owner, fields: chargeAdjustmentFields(record, owner) };
	}
	if (resource === "credit_notes") {
		const number = text(record.number);
		return { ...base, title: `Credit note ${number || id}`, customerId, fields: creditNoteFields(record, customerId) };
	}
	if (resource === "coupons") {
		return { ...base, title: `Coupon ${text(record.name) || id}`, customerId: null, fields: couponFields(record) };
	}
	if (resource === "products") {
		return { ...base, title: `Product ${text(record.name) || id}`, customerId: null, fields: productFields(record) };
	}
	if (resource === "prices") {
		return { ...base, title: `Price ${text(record.nickname) || id}`, customerId: null, fields: priceFields(record) };
	}
	if (resource === "payouts") {
		return { ...base, title: `Payout ${id}`, customerId: null, fields: payoutFields(record) };
	}
	if (resource === "balance_transactions") {
		return { ...base, title: `Balance transaction ${id}`, customerId: null, fields: balanceTransactionFields(record) };
	}
	const subject = asRecord(asRecord(record.data).object);
	const eventCustomer = subject.object === "customer" ? text(subject.id) || null : expandableId(subject.customer);
	return {
		...base,
		title: `Event ${text(record.type) || id}`,
		customerId: eventCustomer,
		fields: eventFields(record, subject, eventCustomer),
	};
}

function customerFields(record: Record<string, unknown>): StripeRecordField[] {
	const currency = text(record.currency);
	return compact([
		field("Name", text(record.name)),
		field("Email", text(record.email)),
		field("Phone", text(record.phone)),
		field("Description", text(record.description)),
		field("Created", timestamp(record.created)),
		field("Currency", currency.toUpperCase()),
		field("Delinquent", bool(record.delinquent)),
		field("Balance", currency ? money(record.balance, currency) : ""),
		...metadataFields(record.metadata),
	]);
}

function subscriptionFields(record: Record<string, unknown>, customerId: string | null): StripeRecordField[] {
	const items = asRecord(record.items);
	const lines = Array.isArray(items.data) ? items.data.map(asRecord) : [];
	const periodEnd = record.current_period_end ?? lines[0]?.current_period_end;
	const periodStart = record.current_period_start ?? lines[0]?.current_period_start;
	return compact([
		field("Customer", customerId ?? ""),
		field("Status", text(record.status)),
		field("Created", timestamp(record.created)),
		field("Start date", timestamp(record.start_date)),
		field("Current period start", timestamp(periodStart)),
		field("Current period end", timestamp(periodEnd)),
		field("Cancel at period end", bool(record.cancel_at_period_end)),
		field("Cancel at", timestamp(record.cancel_at)),
		field("Canceled at", timestamp(record.canceled_at)),
		field("Ended at", timestamp(record.ended_at)),
		field("Trial end", timestamp(record.trial_end)),
		field("Collection method", text(record.collection_method)),
		field("Latest invoice", expandableId(record.latest_invoice) ?? ""),
		field("Discounts", discountIds(record.discounts)),
		...lines.slice(0, MAX_LINE_ITEMS).map((line, index) => field(`Item ${index + 1}`, subscriptionItem(line))),
		field("Items not shown", lines.length > MAX_LINE_ITEMS ? String(lines.length - MAX_LINE_ITEMS) : ""),
		...metadataFields(record.metadata),
	]);
}

function subscriptionItem(line: Record<string, unknown>): string {
	const price = asRecord(line.price);
	const recurring = asRecord(price.recurring);
	const currency = text(price.currency);
	const amount = currency ? money(price.unit_amount, currency) : "";
	const interval = text(recurring.interval);
	const count =
		typeof recurring.interval_count === "number" && recurring.interval_count > 1 ? recurring.interval_count : 1;
	const cadence = interval ? `per ${count > 1 ? `${count} ${interval}s` : interval}` : "";
	const quantity = typeof line.quantity === "number" ? `quantity ${line.quantity}` : "";
	const product = expandableId(price.product);
	return [text(price.id), product ? `product ${product}` : "", [amount, cadence].filter(Boolean).join(" "), quantity]
		.filter(Boolean)
		.join(", ");
}

function invoiceFields(record: Record<string, unknown>, customerId: string | null): StripeRecordField[] {
	const currency = text(record.currency);
	const parent = asRecord(asRecord(record.parent).subscription_details);
	const subscription = expandableId(record.subscription) ?? expandableId(parent.subscription);
	const amount = (value: unknown): string => (currency ? money(value, currency) : "");
	const lineList = asRecord(record.lines);
	const invoiceLines = Array.isArray(lineList.data) ? lineList.data.map(asRecord) : [];
	const invoiceLineTotal =
		typeof lineList.total_count === "number"
			? lineList.total_count
			: invoiceLines.length + (lineList.has_more === true ? 1 : 0);
	const invoiceLinesShown = Math.min(invoiceLines.length, MAX_LINE_ITEMS);
	return compact([
		field("Number", text(record.number)),
		field("Customer", customerId ?? ""),
		field("Customer name", text(record.customer_name)),
		field("Customer email", text(record.customer_email)),
		field("Status", text(record.status)),
		field("Total", amount(record.total)),
		field("Amount due", amount(record.amount_due)),
		field("Amount paid", amount(record.amount_paid)),
		field("Amount remaining", amount(record.amount_remaining)),
		field("Created", timestamp(record.created)),
		field("Due date", timestamp(record.due_date)),
		field("Period start", timestamp(record.period_start)),
		field("Period end", timestamp(record.period_end)),
		field("Subscription", subscription ?? ""),
		field("Collection method", text(record.collection_method)),
		field("Attempt count", typeof record.attempt_count === "number" ? String(record.attempt_count) : ""),
		field("Next payment attempt", timestamp(record.next_payment_attempt)),
		field("Description", text(record.description)),
		field("Discounts", discountIds(record.discounts)),
		...invoiceLines
			.slice(0, MAX_LINE_ITEMS)
			.map((line, index) =>
				field(
					`Line ${index + 1}`,
					[text(line.description), currency ? money(line.amount, currency) : "", periodText(line.period)]
						.filter(Boolean)
						.join(" · "),
				),
			),
		field("Lines not shown", invoiceLineTotal > invoiceLinesShown ? String(invoiceLineTotal - invoiceLinesShown) : ""),
		...metadataFields(record.metadata),
	]);
}

function chargeFields(record: Record<string, unknown>, customerId: string | null): StripeRecordField[] {
	const currency = text(record.currency);
	const amount = (value: unknown): string => (currency ? money(value, currency) : "");
	return compact([
		field("Customer", customerId ?? ""),
		field("Amount", amount(record.amount)),
		field("Amount refunded", record.amount_refunded === 0 ? "" : amount(record.amount_refunded)),
		field("Status", text(record.status)),
		field("Paid", bool(record.paid)),
		field("Refunded", bool(record.refunded)),
		field("Disputed", bool(record.disputed)),
		field("Failure code", text(record.failure_code)),
		field("Failure message", text(record.failure_message)),
		field("Created", timestamp(record.created)),
		field("Description", text(record.description)),
		field("Payment method type", text(asRecord(record.payment_method_details).type)),
		field("Payment intent", expandableId(record.payment_intent) ?? ""),
		...metadataFields(record.metadata),
	]);
}

function discountIds(value: unknown): string {
	if (!Array.isArray(value)) return "";
	return value
		.map((entry) => {
			if (typeof entry === "string") return entry;
			const discount = asRecord(entry);
			const coupon = asRecord(discount.coupon);
			return [text(coupon.name) || text(coupon.id), text(discount.id)].filter(Boolean).join(" ");
		})
		.filter(Boolean)
		.slice(0, MAX_LINE_ITEMS)
		.join(", ");
}

function periodText(value: unknown): string {
	const period = asRecord(value);
	const start = timestamp(period.start);
	const end = timestamp(period.end);
	return start && end ? `${start} to ${end}` : "";
}

function amountOf(record: Record<string, unknown>, key: string): string {
	const currency = text(record.currency);
	return currency ? money(record[key], currency) : "";
}

function chargeAdjustmentFields(record: Record<string, unknown>, customerId: string | null): StripeRecordField[] {
	return compact([
		field("Customer", customerId ?? ""),
		field("Charge", expandableId(record.charge) ?? ""),
		field("Payment intent", expandableId(record.payment_intent) ?? ""),
		field("Amount", amountOf(record, "amount")),
		field("Status", text(record.status)),
		field("Reason", text(record.reason)),
		field("Failure reason", text(record.failure_reason)),
		field("Created", timestamp(record.created)),
		field("Evidence due by", timestamp(asRecord(record.evidence_details).due_by)),
		field("Description", text(record.description)),
		...metadataFields(record.metadata),
	]);
}

function creditNoteFields(record: Record<string, unknown>, customerId: string | null): StripeRecordField[] {
	return compact([
		field("Number", text(record.number)),
		field("Customer", customerId ?? ""),
		field("Invoice", expandableId(record.invoice) ?? ""),
		field("Status", text(record.status)),
		field("Type", text(record.type)),
		field("Reason", text(record.reason)),
		field("Total", amountOf(record, "total")),
		field("Created", timestamp(record.created)),
		field("Memo", text(record.memo)),
		...metadataFields(record.metadata),
	]);
}

function couponFields(record: Record<string, unknown>): StripeRecordField[] {
	return compact([
		field("Name", text(record.name)),
		field("Percent off", typeof record.percent_off === "number" ? `${record.percent_off}%` : ""),
		field("Amount off", record.amount_off == null ? "" : amountOf(record, "amount_off")),
		field("Duration", text(record.duration)),
		field("Duration in months", typeof record.duration_in_months === "number" ? String(record.duration_in_months) : ""),
		field("Times redeemed", typeof record.times_redeemed === "number" ? String(record.times_redeemed) : ""),
		field("Max redemptions", typeof record.max_redemptions === "number" ? String(record.max_redemptions) : ""),
		field("Redeem by", timestamp(record.redeem_by)),
		field("Valid", bool(record.valid)),
		field("Created", timestamp(record.created)),
		...metadataFields(record.metadata),
	]);
}

function productFields(record: Record<string, unknown>): StripeRecordField[] {
	return compact([
		field("Name", text(record.name)),
		field("Active", bool(record.active)),
		field("Description", text(record.description)),
		field("Default price", expandableId(record.default_price) ?? ""),
		field("Created", timestamp(record.created)),
		...metadataFields(record.metadata),
	]);
}

function priceFields(record: Record<string, unknown>): StripeRecordField[] {
	const recurring = asRecord(record.recurring);
	const interval = text(recurring.interval);
	const count = typeof recurring.interval_count === "number" ? recurring.interval_count : 1;
	return compact([
		field("Nickname", text(record.nickname)),
		field("Product", expandableId(record.product) ?? ""),
		field("Active", bool(record.active)),
		field("Unit amount", record.unit_amount == null ? "" : amountOf(record, "unit_amount")),
		field("Type", text(record.type)),
		field("Billing interval", interval ? `every ${count} ${interval}` : ""),
		field("Created", timestamp(record.created)),
		...metadataFields(record.metadata),
	]);
}

function payoutFields(record: Record<string, unknown>): StripeRecordField[] {
	return compact([
		field("Amount", amountOf(record, "amount")),
		field("Status", text(record.status)),
		field("Type", text(record.type)),
		field("Method", text(record.method)),
		field("Created", timestamp(record.created)),
		field("Arrival date", timestamp(record.arrival_date)),
		field("Failure code", text(record.failure_code)),
		field("Description", text(record.description)),
	]);
}

function balanceTransactionFields(record: Record<string, unknown>): StripeRecordField[] {
	return compact([
		field("Type", text(record.type)),
		field("Reporting category", text(record.reporting_category)),
		field("Source", expandableId(record.source) ?? ""),
		field("Amount", amountOf(record, "amount")),
		field("Fee", amountOf(record, "fee")),
		field("Net", amountOf(record, "net")),
		field("Status", text(record.status)),
		field("Created", timestamp(record.created)),
		field("Available on", timestamp(record.available_on)),
		field("Description", text(record.description)),
	]);
}

function eventFields(
	record: Record<string, unknown>,
	subject: Record<string, unknown>,
	customerId: string | null,
): StripeRecordField[] {
	const changed = Object.keys(asRecord(asRecord(record.data).previous_attributes)).sort();
	return compact([
		field("Type", text(record.type)),
		field("Customer", customerId ?? ""),
		field("Object", [text(subject.object), text(subject.id)].filter(Boolean).join(" ")),
		field("Object status", text(subject.status)),
		field("Changed fields", changed.slice(0, 20).join(", ")),
		field("Created", timestamp(record.created)),
	]);
}

function metadataFields(value: unknown): StripeRecordField[] {
	const entries = Object.entries(asRecord(value)).filter(
		(entry): entry is [string, string] => typeof entry[1] === "string" && entry[1].length > 0,
	);
	const fields = entries
		.sort(([left], [right]) => left.localeCompare(right))
		.slice(0, MAX_METADATA_ENTRIES)
		.map(([key, entry]) => field(`Metadata ${clip(key, 80)}`, entry));
	if (entries.length > MAX_METADATA_ENTRIES)
		fields.push(field("Metadata entries not shown", String(entries.length - MAX_METADATA_ENTRIES)));
	return fields;
}

function field(name: string, value: string): StripeRecordField {
	return { name, value: clip(value.replace(/\s+/g, " ").trim(), MAX_FIELD_VALUE_CHARS) };
}

function compact(fields: readonly StripeRecordField[]): StripeRecordField[] {
	return fields.filter((entry) => entry.value.length > 0);
}

function clip(value: string, max: number): string {
	return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

function text(value: unknown): string {
	return typeof value === "string" ? value : "";
}

function bool(value: unknown): string {
	return typeof value === "boolean" ? (value ? "yes" : "no") : "";
}

function timestamp(value: unknown): string {
	if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return "";
	const date = new Date(value * 1_000);
	return Number.isNaN(date.getTime()) ? "" : date.toISOString();
}

export function money(value: unknown, currency: string): string {
	if (typeof value !== "number" || !Number.isFinite(value)) return "";
	const code = currency.toLowerCase();
	const decimals = ZERO_DECIMAL_CURRENCIES.has(code) ? 0 : THREE_DECIMAL_CURRENCIES.has(code) ? 3 : 2;
	return `${(value / 10 ** decimals).toFixed(decimals)} ${code.toUpperCase()}`;
}

function expandableId(value: unknown): string | null {
	if (typeof value === "string" && value) return value;
	const id = asRecord(value).id;
	return typeof id === "string" && id ? id : null;
}

function asRecord(value: unknown): Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}
