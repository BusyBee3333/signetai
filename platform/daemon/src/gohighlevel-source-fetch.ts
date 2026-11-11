import type { GoHighLevelSourceResourceType } from "@signet/core";
import { htmlToText } from "./source-html-text";

export const GOHIGHLEVEL_API_BASE = "https://services.leadconnectorhq.com";
export const GOHIGHLEVEL_API_VERSION = "2021-07-28";
export const GOHIGHLEVEL_USER_AGENT = "Signet-Source/1.0";
export const GOHIGHLEVEL_MAX_RESPONSE_BYTES = 16 * 1024 * 1024;
export const GOHIGHLEVEL_PAGE_SIZE = 100;
const REQUEST_TIMEOUT_MS = 30_000;
const MAX_ATTEMPTS = 4;
const RETRY_BASE_DELAY_MS = 1_000;
const MAX_RETRY_DELAY_MS = 60_000;
const MAX_FIELD_VALUE_CHARS = 2_000;
const MAX_CUSTOM_FIELDS = 25;
const MAX_TAGS = 50;
export const GOHIGHLEVEL_MAX_CONVERSATION_MESSAGES = 300;
export const GOHIGHLEVEL_MAX_FULL_EMAILS_PER_CONVERSATION = 20;
export const GOHIGHLEVEL_MAX_TRANSCRIPT_CHARS = 100_000;
const MAX_MESSAGE_BODY_CHARS = 4_000;
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

const CANCEL_POLL_MS = 250;

type Sleep = (ms: number, shouldContinue: () => boolean) => Promise<void>;
type ShouldContinue = () => boolean;

let goHighLevelSleep: Sleep = cancellableSleep;

export function setGoHighLevelSleepForTest(sleep: Sleep | null): void {
	goHighLevelSleep = sleep ?? cancellableSleep;
}

async function cancellableSleep(ms: number, shouldContinue: ShouldContinue): Promise<void> {
	const deadline = Date.now() + ms;
	while (Date.now() < deadline && shouldContinue()) {
		await new Promise((resolve) => setTimeout(resolve, Math.min(CANCEL_POLL_MS, deadline - Date.now())));
	}
}

export class GoHighLevelRequestError extends Error {
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
		this.name = "GoHighLevelRequestError";
		this.status = options.status;
		this.code = options.code;
		this.retryable = options.retryable;
		this.retryAfterMs = options.retryAfterMs ?? null;
	}
}

export interface GoHighLevelRecordField {
	readonly name: string;
	readonly value: string;
}

export interface GoHighLevelRecord {
	readonly resource: GoHighLevelSourceResourceType;
	readonly id: string;
	readonly title: string;
	readonly createdMs: number | null;
	readonly updatedMs: number | null;
	readonly contactId: string | null;
	readonly fields: readonly GoHighLevelRecordField[];
	readonly versionKey: string | null;
}

export interface GoHighLevelTranscript {
	readonly text: string;
	readonly messages: number;
	readonly olderMessagesOmitted: boolean;
	readonly clipped: boolean;
	readonly emailPreviewsOnly: number;
}

export interface GoHighLevelCursor {
	readonly startAfter: number;
	readonly startAfterId: string;
}

export interface GoHighLevelListPage {
	readonly records: readonly GoHighLevelRecord[];
	readonly unparsed: number;
	readonly total: number | null;
	readonly next: GoHighLevelCursor | null;
}

export async function listGoHighLevelPage(
	token: string,
	locationId: string,
	resource: GoHighLevelSourceResourceType,
	cursor: GoHighLevelCursor | null,
	limit: number,
	shouldContinue: ShouldContinue,
): Promise<GoHighLevelListPage> {
	const pageSize = Math.max(1, Math.min(GOHIGHLEVEL_PAGE_SIZE, limit));
	const query = new URLSearchParams({ limit: String(pageSize) });
	query.set(resource === "opportunities" ? "location_id" : "locationId", locationId);
	if (cursor && resource === "conversations") {
		query.set("startAfterDate", String(cursor.startAfter));
	} else if (cursor) {
		query.set("startAfter", String(cursor.startAfter));
		query.set("startAfterId", cursor.startAfterId);
	}
	const path =
		resource === "contacts"
			? "/contacts/"
			: resource === "opportunities"
				? "/opportunities/search"
				: "/conversations/search";
	const body = asRecord(await goHighLevelRequest(token, `${path}?${query.toString()}`, shouldContinue));
	const items = body[resource];
	if (!Array.isArray(items)) {
		throw new GoHighLevelRequestError(`GoHighLevel list response has no ${resource} array`, {
			status: 200,
			code: "invalid_response",
			retryable: false,
		});
	}
	const records: GoHighLevelRecord[] = [];
	let unparsed = 0;
	for (const item of items) {
		const record = parseGoHighLevelRecord(resource, item);
		if (record) records.push(record);
		else unparsed++;
	}
	if (resource === "conversations") {
		const total = typeof body.total === "number" && Number.isFinite(body.total) ? body.total : null;
		if (items.length === 0) return { records, unparsed, total, next: null };
		const last = asRecord(items[items.length - 1]);
		const sort = Array.isArray(last.sort) ? last.sort[0] : undefined;
		const position = typeof sort === "number" ? sort : last.lastMessageDate;
		if (typeof position !== "number" || !Number.isFinite(position)) {
			throw new GoHighLevelRequestError("GoHighLevel returned a conversations page with no cursor for the next page", {
				status: 200,
				code: "invalid_response",
				retryable: false,
			});
		}
		return { records, unparsed, total, next: { startAfter: position, startAfterId: "" } };
	}
	const meta = asRecord(body.meta);
	const total = typeof meta.total === "number" && Number.isFinite(meta.total) ? meta.total : null;
	const next =
		items.length > 0 &&
		typeof meta.startAfter === "number" &&
		typeof meta.startAfterId === "string" &&
		meta.startAfterId
			? { startAfter: meta.startAfter, startAfterId: meta.startAfterId }
			: null;
	return { records, unparsed, total, next };
}

async function goHighLevelRequest(
	token: string,
	pathAndQuery: string,
	shouldContinue: ShouldContinue,
): Promise<unknown> {
	let lastError: GoHighLevelRequestError | null = null;
	for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
		if (lastError) await goHighLevelSleep(retryDelay(lastError, attempt), shouldContinue);
		if (!shouldContinue()) throw cancelledRequest();
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
		const cancelPoll = setInterval(() => {
			if (!shouldContinue()) controller.abort();
		}, CANCEL_POLL_MS);
		try {
			const response = await fetch(`${GOHIGHLEVEL_API_BASE}${pathAndQuery}`, {
				method: "GET",
				headers: {
					Authorization: `Bearer ${token}`,
					Version: GOHIGHLEVEL_API_VERSION,
					Accept: "application/json",
					"User-Agent": GOHIGHLEVEL_USER_AGENT,
				},
				signal: controller.signal,
			});
			const text = await readBoundedText(response);
			if (response.ok) return parseJson(text, response.status);
			const error = responseError(response, parseErrorBody(text));
			if (!error.retryable) throw error;
			lastError = error;
		} catch (err) {
			if (err instanceof GoHighLevelRequestError && !err.retryable) throw err;
			if (!shouldContinue()) throw cancelledRequest();
			lastError =
				err instanceof GoHighLevelRequestError
					? err
					: new GoHighLevelRequestError(
							controller.signal.aborted
								? `GoHighLevel request timed out after ${REQUEST_TIMEOUT_MS}ms`
								: `GoHighLevel request failed: ${err instanceof Error ? err.message : String(err)}`,
							{ status: 0, code: controller.signal.aborted ? "timeout" : "network", retryable: true },
						);
		} finally {
			clearTimeout(timeout);
			clearInterval(cancelPoll);
		}
	}
	throw (
		lastError ??
		new GoHighLevelRequestError("GoHighLevel request failed", { status: 0, code: "network", retryable: true })
	);
}

function responseError(response: Response, body: unknown): GoHighLevelRequestError {
	const record = asRecord(body);
	const code = typeof record.error === "string" ? record.error : `http_${response.status}`;
	const message = Array.isArray(record.message) ? record.message.filter(isString).join("; ") : text(record.message);
	const detail = message || text(record.title) || response.statusText;
	const retryAfter = Number(response.headers.get("retry-after") ?? Number.NaN);
	return new GoHighLevelRequestError(`GoHighLevel API ${response.status} ${code}: ${detail}`.trim(), {
		status: response.status,
		code,
		retryable: RETRYABLE_STATUSES.has(response.status),
		...(Number.isFinite(retryAfter) && retryAfter >= 0 ? { retryAfterMs: retryAfter * 1_000 } : {}),
	});
}

function retryDelay(error: GoHighLevelRequestError, attempt: number): number {
	const backoff = RETRY_BASE_DELAY_MS * 2 ** (attempt - 1);
	return Math.min(MAX_RETRY_DELAY_MS, Math.max(error.retryAfterMs ?? 0, backoff));
}

function cancelledRequest(): GoHighLevelRequestError {
	return new GoHighLevelRequestError("GoHighLevel request cancelled", {
		status: 0,
		code: "cancelled",
		retryable: false,
	});
}

async function readBoundedText(response: Response): Promise<string> {
	const declared = Number(response.headers.get("content-length"));
	if (Number.isFinite(declared) && declared > GOHIGHLEVEL_MAX_RESPONSE_BYTES) {
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
		if (total > GOHIGHLEVEL_MAX_RESPONSE_BYTES) {
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

function parseJson(body: string, status: number): unknown {
	try {
		return JSON.parse(body);
	} catch {
		throw new GoHighLevelRequestError(`GoHighLevel API ${status} returned invalid JSON`, {
			status,
			code: "invalid_response",
			retryable: false,
		});
	}
}

function parseErrorBody(body: string): unknown {
	try {
		return JSON.parse(body);
	} catch {
		return null;
	}
}

function oversizedResponse(status: number): GoHighLevelRequestError {
	return new GoHighLevelRequestError(`GoHighLevel API response exceeded ${GOHIGHLEVEL_MAX_RESPONSE_BYTES} bytes`, {
		status,
		code: "response_too_large",
		retryable: false,
	});
}

export function parseGoHighLevelRecord(
	resource: GoHighLevelSourceResourceType,
	value: unknown,
): GoHighLevelRecord | null {
	const record = asRecord(value);
	const id = text(record.id);
	if (!id) return null;
	if (resource === "contacts") {
		const name = text(record.contactName) || [text(record.firstName), text(record.lastName)].filter(Boolean).join(" ");
		return {
			resource,
			id,
			title: name || text(record.companyName) || text(record.email) || id,
			createdMs: dateMs(record.dateAdded),
			updatedMs: dateMs(record.dateUpdated),
			contactId: id,
			fields: contactFields(record),
			versionKey: null,
		};
	}
	if (resource === "conversations") {
		const name = text(record.contactName) || text(record.fullName);
		return {
			resource,
			id,
			title: `Conversation with ${name || text(record.email) || text(record.phone) || id}`,
			createdMs: dateMs(record.dateAdded),
			updatedMs: dateMs(record.lastMessageDate) ?? dateMs(record.dateUpdated),
			contactId: text(record.contactId) || null,
			fields: conversationFields(record),
			versionKey: `${String(record.lastMessageDate ?? "")}:${String(record.dateUpdated ?? "")}`,
		};
	}
	return {
		resource,
		id,
		title: text(record.name) || `Opportunity ${id}`,
		createdMs: dateMs(record.createdAt),
		updatedMs: dateMs(record.updatedAt),
		contactId: text(record.contactId) || null,
		fields: opportunityFields(record),
		versionKey: null,
	};
}

function conversationFields(record: Record<string, unknown>): GoHighLevelRecordField[] {
	const tags = stringList(record.tags);
	return compact([
		field("Contact", text(record.contactId)),
		field("Contact name", text(record.contactName) || text(record.fullName)),
		field("Company", text(record.companyName)),
		field("Email", text(record.email)),
		field("Phone", text(record.phone)),
		field("Channel", typeLabel(record.type)),
		field("Last message type", typeLabel(record.lastMessageType)),
		field("Last message direction", text(record.lastMessageDirection)),
		field("Last message", dateText(record.lastMessageDate)),
		field("Unread", typeof record.unreadCount === "number" && record.unreadCount > 0 ? String(record.unreadCount) : ""),
		field("Tags", tags.slice(0, MAX_TAGS).join(", ")),
	]);
}

export async function fetchGoHighLevelTranscript(
	token: string,
	conversationId: string,
	shouldContinue: ShouldContinue,
): Promise<GoHighLevelTranscript> {
	const collected: Record<string, unknown>[] = [];
	let lastMessageId: string | null = null;
	let more = true;
	while (more && collected.length < GOHIGHLEVEL_MAX_CONVERSATION_MESSAGES && shouldContinue()) {
		const query = new URLSearchParams({ limit: String(GOHIGHLEVEL_PAGE_SIZE) });
		if (lastMessageId) query.set("lastMessageId", lastMessageId);
		const body = asRecord(
			await goHighLevelRequest(
				token,
				`/conversations/${encodeURIComponent(conversationId)}/messages?${query.toString()}`,
				shouldContinue,
			),
		);
		const page = asRecord(body.messages);
		if (!Array.isArray(page.messages)) {
			throw new GoHighLevelRequestError("GoHighLevel messages response has no messages array", {
				status: 200,
				code: "invalid_response",
				retryable: false,
			});
		}
		for (const item of page.messages) collected.push(asRecord(item));
		const cursor = text(page.lastMessageId);
		more = page.nextPage === true && page.messages.length > 0 && cursor.length > 0 && cursor !== lastMessageId;
		lastMessageId = cursor || lastMessageId;
	}
	const kept = collected.slice(0, GOHIGHLEVEL_MAX_CONVERSATION_MESSAGES);
	const lines: string[] = [];
	let fullEmails = 0;
	let emailPreviewsOnly = 0;
	for (const message of kept) {
		if (!shouldContinue()) throw cancelledRequest();
		const email = asRecord(asRecord(message.meta).email);
		let body = text(message.body);
		let subject = text(email.subject);
		const emailId = stringList(email.messageIds)[0];
		if (typeLabel(message.messageType) === "EMAIL" && emailId) {
			if (fullEmails < GOHIGHLEVEL_MAX_FULL_EMAILS_PER_CONVERSATION) {
				fullEmails++;
				const full = await fetchFullEmail(token, emailId, shouldContinue);
				if (full) {
					body = full.body;
					subject = full.subject || subject;
				} else emailPreviewsOnly++;
			} else emailPreviewsOnly++;
		}
		const rendered = renderBody(body, text(message.contentType));
		if (!rendered && !subject) continue;
		const attachments = Array.isArray(message.attachments) ? message.attachments.length : 0;
		const header = [
			dateText(message.dateAdded) || "undated",
			text(message.direction) || "activity",
			typeLabel(message.messageType).toLowerCase() || "message",
			text(message.status),
		]
			.filter(Boolean)
			.join(" · ");
		lines.push(
			[
				`### ${header}`,
				subject ? `Subject: ${clip(subject.replace(/\s+/g, " ").trim(), 300)}` : "",
				attachments > 0 ? `Attachments: ${attachments}` : "",
				clip(rendered, MAX_MESSAGE_BODY_CHARS),
			]
				.filter(Boolean)
				.join("\n"),
		);
	}
	const full = lines.reverse().join("\n\n");
	return {
		text: full.slice(0, GOHIGHLEVEL_MAX_TRANSCRIPT_CHARS),
		messages: lines.length,
		olderMessagesOmitted: more || collected.length > kept.length,
		clipped: full.length > GOHIGHLEVEL_MAX_TRANSCRIPT_CHARS,
		emailPreviewsOnly,
	};
}

async function fetchFullEmail(
	token: string,
	emailMessageId: string,
	shouldContinue: ShouldContinue,
): Promise<{ readonly subject: string; readonly body: string } | null> {
	try {
		const body = asRecord(
			await goHighLevelRequest(
				token,
				`/conversations/messages/email/${encodeURIComponent(emailMessageId)}`,
				shouldContinue,
			),
		);
		const email = asRecord(body.emailMessage);
		const content = text(email.body);
		return content ? { subject: text(email.subject), body: content } : null;
	} catch (err) {
		if (err instanceof GoHighLevelRequestError && (err.status === 404 || err.status === 400 || err.status === 422)) {
			return null;
		}
		throw err;
	}
}

function renderBody(body: string, contentType: string): string {
	const looksLikeHtml =
		contentType.toLowerCase().includes("html") || /<\/?(html|body|div|p|br|table|span|a)\b/i.test(body);
	return (looksLikeHtml ? htmlToText(body) : body.replace(/\r\n/g, "\n")).trim();
}

function typeLabel(value: unknown): string {
	return text(value)
		.replace(/^TYPE_/, "")
		.replace(/_/g, " ");
}

function contactFields(record: Record<string, unknown>): GoHighLevelRecordField[] {
	const tags = stringList(record.tags);
	const emails = stringList(record.additionalEmails);
	return compact([
		field("First name", text(record.firstName)),
		field("Last name", text(record.lastName)),
		field("Company", text(record.companyName)),
		field("Email", text(record.email)),
		field("Additional emails", emails.slice(0, MAX_TAGS).join(", ")),
		field("Phone", text(record.phone)),
		field("Type", text(record.type)),
		field("Source", text(record.source)),
		field("Tags", tags.slice(0, MAX_TAGS).join(", ")),
		field("Tags not shown", tags.length > MAX_TAGS ? String(tags.length - MAX_TAGS) : ""),
		field("City", text(record.city)),
		field("State", text(record.state)),
		field("Country", text(record.country)),
		field("Timezone", text(record.timezone)),
		field("Website", text(record.website)),
		field("Do not disturb", bool(record.dnd)),
		field("Assigned to", text(record.assignedTo)),
		field("Business", text(record.businessId)),
		field("Added", dateText(record.dateAdded)),
		field("Updated", dateText(record.dateUpdated)),
		...customFields(record.customFields),
	]);
}

function opportunityFields(record: Record<string, unknown>): GoHighLevelRecordField[] {
	const contact = asRecord(record.contact);
	return compact([
		field("Status", text(record.status)),
		field("Value", typeof record.monetaryValue === "number" ? String(record.monetaryValue) : ""),
		field("Pipeline", text(record.pipelineId)),
		field("Pipeline stage", text(record.pipelineStageId)),
		field("Contact", text(record.contactId)),
		field("Contact name", text(contact.name)),
		field("Contact company", text(contact.companyName)),
		field("Contact email", text(contact.email)),
		field("Source", text(record.source)),
		field("Assigned to", text(record.assignedTo)),
		field("Lost reason", text(record.lostReasonId)),
		field("Created", dateText(record.createdAt)),
		field("Updated", dateText(record.updatedAt)),
		field("Last status change", dateText(record.lastStatusChangeAt)),
		field("Last stage change", dateText(record.lastStageChangeAt)),
		...customFields(record.customFields),
	]);
}

function customFields(value: unknown): GoHighLevelRecordField[] {
	const entries = (Array.isArray(value) ? value : [])
		.map(asRecord)
		.map((entry) => ({ id: text(entry.id), value: customFieldValue(entry) }))
		.filter((entry) => entry.id && entry.value)
		.sort((left, right) => left.id.localeCompare(right.id));
	const fields = entries.slice(0, MAX_CUSTOM_FIELDS).map((entry) => field(`Custom field ${entry.id}`, entry.value));
	if (entries.length > MAX_CUSTOM_FIELDS)
		fields.push(field("Custom fields not shown", String(entries.length - MAX_CUSTOM_FIELDS)));
	return fields;
}

function customFieldValue(entry: Record<string, unknown>): string {
	const value = entry.value ?? entry.fieldValue ?? entry.fieldValueString;
	if (typeof value === "string") return value;
	if (typeof value === "number" || typeof value === "boolean") return String(value);
	if (Array.isArray(value)) return value.filter(isString).join(", ");
	return "";
}

function field(name: string, value: string): GoHighLevelRecordField {
	return { name, value: clip(value.replace(/\s+/g, " ").trim(), MAX_FIELD_VALUE_CHARS) };
}

function compact(fields: readonly GoHighLevelRecordField[]): GoHighLevelRecordField[] {
	return fields.filter((entry) => entry.value.length > 0);
}

function clip(value: string, max: number): string {
	return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

function text(value: unknown): string {
	return typeof value === "string" ? value : "";
}

function isString(value: unknown): value is string {
	return typeof value === "string";
}

function stringList(value: unknown): string[] {
	return Array.isArray(value) ? value.filter(isString).filter((entry) => entry.length > 0) : [];
}

function bool(value: unknown): string {
	return typeof value === "boolean" ? (value ? "yes" : "no") : "";
}

function dateMs(value: unknown): number | null {
	const ms = typeof value === "number" ? value : typeof value === "string" ? Date.parse(value) : Number.NaN;
	return Number.isFinite(ms) && ms > 0 ? ms : null;
}

function dateText(value: unknown): string {
	const ms = dateMs(value);
	return ms === null ? "" : new Date(ms).toISOString();
}

function asRecord(value: unknown): Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}
