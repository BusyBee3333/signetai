import { htmlToText } from "./source-html-text";

export const GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me";
export const GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token";
export const GMAIL_MAX_RESPONSE_BYTES = 48 * 1024 * 1024;
export const GMAIL_LIST_PAGE_SIZE = 500;
export const GMAIL_MAX_BODY_CHARS = 100_000;
const REQUEST_TIMEOUT_MS = 30_000;
const MAX_ATTEMPTS = 4;
const RETRY_BASE_DELAY_MS = 1_000;
const MAX_RETRY_DELAY_MS = 60_000;
const MAX_HEADER_VALUE_CHARS = 2_000;
const MAX_ATTACHMENT_NAMES = 25;
const MAX_PART_DEPTH = 12;
const TOKEN_EXPIRY_MARGIN_MS = 60_000;
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);
const RETRYABLE_FORBIDDEN_REASONS = new Set(["rateLimitExceeded", "userRateLimitExceeded"]);

const CANCEL_POLL_MS = 250;

type Sleep = (ms: number, shouldContinue: () => boolean) => Promise<void>;
type ShouldContinue = () => boolean;

let gmailSleep: Sleep = cancellableSleep;

export function setGmailSleepForTest(sleep: Sleep | null): void {
	gmailSleep = sleep ?? cancellableSleep;
}

async function cancellableSleep(ms: number, shouldContinue: ShouldContinue): Promise<void> {
	const deadline = Date.now() + ms;
	while (Date.now() < deadline && shouldContinue()) {
		await new Promise((resolve) => setTimeout(resolve, Math.min(CANCEL_POLL_MS, deadline - Date.now())));
	}
}

export class GmailRequestError extends Error {
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
		this.name = "GmailRequestError";
		this.status = options.status;
		this.code = options.code;
		this.retryable = options.retryable;
		this.retryAfterMs = options.retryAfterMs ?? null;
	}
}

export interface GmailCredentials {
	readonly clientId: string;
	readonly clientSecret: string;
	readonly refreshToken: string;
}

export interface GmailAccessToken {
	readonly token: string;
	readonly expiresAtMs: number;
}

export interface GmailMessageRef {
	readonly id: string;
	readonly threadId: string | null;
}

export interface GmailMessageList {
	readonly messages: readonly GmailMessageRef[];
	readonly nextPageToken: string | null;
}

export interface GmailMessage {
	readonly id: string;
	readonly threadId: string | null;
	readonly subject: string;
	readonly from: string;
	readonly to: string;
	readonly cc: string;
	readonly sentAtMs: number | null;
	readonly body: string;
	readonly bodyClipped: boolean;
	readonly attachmentNames: readonly string[];
	readonly attachmentCount: number;
}

export type GmailMessageLookup =
	| { readonly status: "found"; readonly message: GmailMessage }
	| { readonly status: "gone" };

export function parseGmailCredentials(secret: string): GmailCredentials {
	let parsed: unknown;
	try {
		parsed = JSON.parse(secret);
	} catch {
		throw invalidCredentials("is not JSON");
	}
	const record = asRecord(parsed);
	const clientId = text(record.client_id).trim();
	const clientSecret = text(record.client_secret).trim();
	const refreshToken = text(record.refresh_token).trim();
	if (!clientId || !clientSecret || !refreshToken) {
		throw invalidCredentials("must contain client_id, client_secret, and refresh_token");
	}
	return { clientId, clientSecret, refreshToken };
}

function invalidCredentials(reason: string): GmailRequestError {
	return new GmailRequestError(`Gmail credentials secret ${reason}`, {
		status: 0,
		code: "invalid_credentials",
		retryable: false,
	});
}

export function gmailTokenIsFresh(token: GmailAccessToken | null, now = Date.now()): token is GmailAccessToken {
	return token !== null && token.expiresAtMs - TOKEN_EXPIRY_MARGIN_MS > now;
}

export async function fetchGmailAccessToken(
	credentials: GmailCredentials,
	shouldContinue: ShouldContinue,
): Promise<GmailAccessToken> {
	const requestedAt = Date.now();
	const body = asRecord(
		await gmailRequest(
			GOOGLE_TOKEN_URL,
			{
				method: "POST",
				headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
				body: new URLSearchParams({
					client_id: credentials.clientId,
					client_secret: credentials.clientSecret,
					refresh_token: credentials.refreshToken,
					grant_type: "refresh_token",
				}).toString(),
			},
			shouldContinue,
		),
	);
	const token = text(body.access_token);
	const expiresIn = typeof body.expires_in === "number" && body.expires_in > 0 ? body.expires_in : 0;
	if (!token || expiresIn === 0) {
		throw new GmailRequestError("Google token response has no usable access token", {
			status: 200,
			code: "invalid_response",
			retryable: false,
		});
	}
	return { token, expiresAtMs: requestedAt + expiresIn * 1_000 };
}

export async function listGmailMessages(
	token: string,
	query: string,
	pageToken: string | null,
	limit: number,
	shouldContinue: ShouldContinue,
): Promise<GmailMessageList> {
	const params = new URLSearchParams({ maxResults: String(Math.max(1, Math.min(GMAIL_LIST_PAGE_SIZE, limit))) });
	if (query) params.set("q", query);
	if (pageToken) params.set("pageToken", pageToken);
	const body = asRecord(await gmailApiRequest(token, `/messages?${params.toString()}`, shouldContinue));
	if (body.messages !== undefined && !Array.isArray(body.messages)) {
		throw new GmailRequestError("Gmail list response has a malformed messages field", {
			status: 200,
			code: "invalid_response",
			retryable: false,
		});
	}
	const messages: GmailMessageRef[] = [];
	for (const item of Array.isArray(body.messages) ? body.messages : []) {
		const record = asRecord(item);
		const id = text(record.id);
		if (!id) {
			throw new GmailRequestError("Gmail list response contains a message without an id", {
				status: 200,
				code: "invalid_response",
				retryable: false,
			});
		}
		messages.push({ id, threadId: text(record.threadId) || null });
	}
	return { messages, nextPageToken: text(body.nextPageToken) || null };
}

export async function fetchGmailMessage(
	token: string,
	id: string,
	shouldContinue: ShouldContinue,
): Promise<GmailMessageLookup> {
	try {
		const body = await gmailApiRequest(token, `/messages/${encodeURIComponent(id)}?format=full`, shouldContinue);
		const message = parseGmailMessage(body);
		if (!message) {
			throw new GmailRequestError(`Gmail message ${id} has an unexpected shape`, {
				status: 200,
				code: "invalid_response",
				retryable: false,
			});
		}
		return { status: "found", message };
	} catch (err) {
		if (err instanceof GmailRequestError && err.status === 404) return { status: "gone" };
		throw err;
	}
}

async function gmailApiRequest(token: string, pathAndQuery: string, shouldContinue: ShouldContinue): Promise<unknown> {
	return await gmailRequest(
		`${GMAIL_API_BASE}${pathAndQuery}`,
		{ method: "GET", headers: { Authorization: `Bearer ${token}`, Accept: "application/json" } },
		shouldContinue,
	);
}

async function gmailRequest(
	url: string,
	init: { readonly method: "GET" | "POST"; readonly headers: Record<string, string>; readonly body?: string },
	shouldContinue: ShouldContinue,
): Promise<unknown> {
	let lastError: GmailRequestError | null = null;
	for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
		if (lastError) await gmailSleep(retryDelay(lastError, attempt), shouldContinue);
		if (!shouldContinue()) throw cancelledRequest();
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
		const cancelPoll = setInterval(() => {
			if (!shouldContinue()) controller.abort();
		}, CANCEL_POLL_MS);
		try {
			const response = await fetch(url, { ...init, signal: controller.signal });
			const body = await readBoundedText(response);
			if (response.ok) return parseJson(body, response.status);
			const error = responseError(response, parseErrorBody(body));
			if (!error.retryable) throw error;
			lastError = error;
		} catch (err) {
			if (err instanceof GmailRequestError && !err.retryable) throw err;
			if (!shouldContinue()) throw cancelledRequest();
			lastError =
				err instanceof GmailRequestError
					? err
					: new GmailRequestError(
							controller.signal.aborted
								? `Gmail request timed out after ${REQUEST_TIMEOUT_MS}ms`
								: `Gmail request failed: ${err instanceof Error ? err.message : String(err)}`,
							{ status: 0, code: controller.signal.aborted ? "timeout" : "network", retryable: true },
						);
		} finally {
			clearTimeout(timeout);
			clearInterval(cancelPoll);
		}
	}
	throw lastError ?? new GmailRequestError("Gmail request failed", { status: 0, code: "network", retryable: true });
}

function responseError(response: Response, body: unknown): GmailRequestError {
	const record = asRecord(body);
	const nested = asRecord(record.error);
	const reasons = (Array.isArray(nested.errors) ? nested.errors : []).map((entry) => text(asRecord(entry).reason));
	const code = text(record.error) || reasons.find(Boolean) || text(nested.status) || `http_${response.status}`;
	const detail = text(record.error_description) || text(nested.message) || response.statusText;
	const retryAfter = Number(response.headers.get("retry-after") ?? Number.NaN);
	const rateLimited = response.status === 403 && reasons.some((reason) => RETRYABLE_FORBIDDEN_REASONS.has(reason));
	return new GmailRequestError(`Gmail API ${response.status} ${code}: ${detail}`.trim(), {
		status: response.status,
		code,
		retryable: rateLimited || RETRYABLE_STATUSES.has(response.status),
		...(Number.isFinite(retryAfter) && retryAfter >= 0 ? { retryAfterMs: retryAfter * 1_000 } : {}),
	});
}

function retryDelay(error: GmailRequestError, attempt: number): number {
	const backoff = RETRY_BASE_DELAY_MS * 2 ** (attempt - 1);
	return Math.min(MAX_RETRY_DELAY_MS, Math.max(error.retryAfterMs ?? 0, backoff));
}

function cancelledRequest(): GmailRequestError {
	return new GmailRequestError("Gmail request cancelled", { status: 0, code: "cancelled", retryable: false });
}

async function readBoundedText(response: Response): Promise<string> {
	const declared = Number(response.headers.get("content-length"));
	if (Number.isFinite(declared) && declared > GMAIL_MAX_RESPONSE_BYTES) {
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
		if (total > GMAIL_MAX_RESPONSE_BYTES) {
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
		throw new GmailRequestError(`Gmail API ${status} returned invalid JSON`, {
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

function oversizedResponse(status: number): GmailRequestError {
	return new GmailRequestError(`Gmail API response exceeded ${GMAIL_MAX_RESPONSE_BYTES} bytes`, {
		status,
		code: "response_too_large",
		retryable: false,
	});
}

export function parseGmailMessage(value: unknown): GmailMessage | null {
	const record = asRecord(value);
	const id = text(record.id);
	if (!id) return null;
	const payload = asRecord(record.payload);
	const headers = new Map<string, string>();
	for (const entry of Array.isArray(payload.headers) ? payload.headers : []) {
		const header = asRecord(entry);
		const name = text(header.name).toLowerCase();
		if (name && !headers.has(name)) headers.set(name, clip(oneLine(text(header.value)), MAX_HEADER_VALUE_CHARS));
	}
	const parts = collectParts(payload);
	const plain = parts.plain.join("\n\n").trim();
	const fullBody = plain || htmlToText(parts.html.join("\n\n"));
	const internalDate = Number(text(record.internalDate) || Number.NaN);
	return {
		id,
		threadId: text(record.threadId) || null,
		subject: headers.get("subject") ?? "",
		from: headers.get("from") ?? "",
		to: headers.get("to") ?? "",
		cc: headers.get("cc") ?? "",
		sentAtMs: Number.isFinite(internalDate) && internalDate > 0 ? internalDate : null,
		body: fullBody.slice(0, GMAIL_MAX_BODY_CHARS),
		bodyClipped: fullBody.length > GMAIL_MAX_BODY_CHARS,
		attachmentNames: parts.attachments.slice(0, MAX_ATTACHMENT_NAMES),
		attachmentCount: parts.attachments.length,
	};
}

interface CollectedParts {
	readonly plain: string[];
	readonly html: string[];
	readonly attachments: string[];
}

function collectParts(payload: Record<string, unknown>): CollectedParts {
	const collected: CollectedParts = { plain: [], html: [], attachments: [] };
	const visit = (part: Record<string, unknown>, depth: number): void => {
		if (depth > MAX_PART_DEPTH) return;
		const filename = text(part.filename).trim();
		const mimeType = text(part.mimeType).toLowerCase();
		const body = asRecord(part.body);
		if (filename) {
			collected.attachments.push(clip(oneLine(filename), 200));
		} else if (mimeType === "text/plain" || mimeType === "text/html") {
			const decoded = decodeBody(text(body.data));
			if (decoded) (mimeType === "text/plain" ? collected.plain : collected.html).push(decoded);
		}
		for (const child of Array.isArray(part.parts) ? part.parts : []) visit(asRecord(child), depth + 1);
	};
	visit(payload, 0);
	return collected;
}

function decodeBody(data: string): string {
	if (!data) return "";
	return Buffer.from(data, "base64url").toString("utf8").replace(/\r\n/g, "\n");
}

function oneLine(value: string): string {
	return value.replace(/\s+/g, " ").trim();
}

function clip(value: string, max: number): string {
	return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

function text(value: unknown): string {
	return typeof value === "string" ? value : "";
}

function asRecord(value: unknown): Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}
