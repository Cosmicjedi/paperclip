import type { AdapterExecutionContext } from "@paperclipai/adapter-utils";
import { asString, buildPaperclipEnv } from "@paperclipai/adapter-utils/server-utils";
import {
  DEFAULT_APPROVAL_BOARD_TIMEOUT_SEC,
  DEFAULT_APPROVAL_POLICY,
} from "../shared/constants.js";

// Hermes parks an API-server run in `waiting_for_approval` when the agent runs a
// command it flags as dangerous, and waits for the client to answer through
// POST /v1/runs/{run_id}/approval. Nobody else is attached to an unattended
// Paperclip run, so this adapter must answer or the command blocks until the
// Hermes approval timeout and the task stalls.

export type ApprovalPolicy = "auto_approve" | "deny" | "ask_board";
export type ApprovalChoice = "once" | "session" | "always" | "deny";

export type HermesApprovalRequest = {
  command: string | null;
  description: string | null;
  patternKeys: string[];
  choices: ApprovalChoice[];
  requestId: string | null;
  smartDenied: boolean;
};

export type ApprovalDecision = {
  choice: ApprovalChoice;
  reason: "policy_auto_approve" | "policy_deny" | "denylist" | "board_accepted" | "board_rejected" | "board_timeout" | "board_unavailable";
  detail?: string;
};

const APPROVAL_CHOICES: readonly ApprovalChoice[] = ["once", "session", "always", "deny"];
const APPROVE_ALIASES = new Set(["approve", "approved", "allow"]);
const BOARD_POLL_INTERVAL_MS = 2_000;
export const APPROVAL_KEEPALIVE_MS = 20_000;

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function nonEmpty(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

function normalizeChoice(value: unknown): ApprovalChoice | null {
  const raw = nonEmpty(value)?.toLowerCase();
  if (!raw) return null;
  if (APPROVE_ALIASES.has(raw)) return "once";
  return (APPROVAL_CHOICES as readonly string[]).includes(raw) ? (raw as ApprovalChoice) : null;
}

export function normalizeApprovalPolicy(value: unknown): ApprovalPolicy {
  const raw = asString(value, DEFAULT_APPROVAL_POLICY).trim().toLowerCase();
  if (raw === "deny" || raw === "ask_board") return raw;
  return "auto_approve";
}

/**
 * Reads the pending approval from a GET /v1/runs/{id} body (its `approval`
 * field) or from an `approval.request` SSE event payload.
 */
export function parseApprovalRequest(value: unknown): HermesApprovalRequest | null {
  const record = asRecord(value);
  if (!record) return null;
  const source = asRecord(record.approval) ?? asRecord(record.data) ?? record;
  const command = nonEmpty(source.command);
  const description = nonEmpty(source.description);
  const requestId = nonEmpty(source.request_id) ?? nonEmpty(source.requestId);
  if (!command && !description && !requestId) return null;
  const patternKeys = [
    ...(Array.isArray(source.pattern_keys) ? source.pattern_keys : []),
    source.pattern_key,
  ].filter((key): key is string => typeof key === "string" && key.length > 0);
  const choices = Array.isArray(source.choices)
    ? source.choices.map(normalizeChoice).filter((choice): choice is ApprovalChoice => choice !== null)
    : [];
  return {
    command,
    description,
    patternKeys: [...new Set(patternKeys)],
    choices,
    requestId,
    smartDenied: source.smart_denied === true,
  };
}

/**
 * Compiles the per-agent denylist: a newline-separated string or an array of
 * regular expressions. Invalid entries are reported and skipped.
 */
export function compileDenylist(value: unknown): { patterns: RegExp[]; invalid: string[] } {
  const entries = Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : typeof value === "string"
      ? value.split(/\r?\n/)
      : [];
  const patterns: RegExp[] = [];
  const invalid: string[] = [];
  for (const entry of entries.map((line) => line.trim()).filter(Boolean)) {
    try {
      patterns.push(new RegExp(entry));
    } catch {
      invalid.push(entry);
    }
  }
  return { patterns, invalid };
}

/** Picks the reply Hermes allows for this request. "deny" is always allowed. */
export function allowedChoice(preferred: ApprovalChoice, request: HermesApprovalRequest): ApprovalChoice {
  if (preferred === "deny") return "deny";
  if (request.choices.length === 0 || request.choices.includes(preferred)) return preferred;
  return "once";
}

export function approvalDedupeKey(request: HermesApprovalRequest, episode: number): string {
  if (request.requestId) return `id:${request.requestId}`;
  return `episode:${episode}:${request.command ?? ""}:${request.description ?? ""}:${request.patternKeys.join(",")}`;
}

type BoardInteraction = { id?: unknown; status?: unknown };

function boardApiBase(ctx: AdapterExecutionContext): string {
  return buildPaperclipEnv(ctx.agent).PAPERCLIP_API_URL.replace(/\/+$/, "");
}

function issueIdFromContext(ctx: AdapterExecutionContext): string | null {
  return nonEmpty(ctx.context.taskId) ?? nonEmpty(ctx.context.issueId);
}

function boardHeaders(ctx: AdapterExecutionContext): Record<string, string> {
  return {
    Authorization: `Bearer ${ctx.authToken}`,
    "Content-Type": "application/json",
    Accept: "application/json",
    "X-Paperclip-Run-Id": ctx.runId,
  };
}

function boardDetails(input: { hermesRunId: string; request: HermesApprovalRequest; redact: (value: string) => string }): string {
  const fence = "```";
  const lines = [
    `Hermes run \`${input.hermesRunId}\` wants to run a command it flagged as dangerous.`,
    "",
    ...(input.request.description ? [`**Why it was flagged:** ${input.redact(input.request.description)}`, ""] : []),
    ...(input.request.patternKeys.length ? [`**Patterns:** ${input.request.patternKeys.join(", ")}`, ""] : []),
    "**Command:**",
    `${fence}sh`,
    input.redact(input.request.command ?? "(command not provided)"),
    fence,
    "",
    "Approve runs it once. Deny, or no answer before the timeout, blocks it.",
  ];
  return lines.join("\n").slice(0, 20_000);
}

/**
 * Asks the board with an issue-thread `request_confirmation` card, the same
 * mechanism agents use to ask the board, and waits for a human answer. No
 * answer before the timeout withdraws the card and denies.
 */
export async function askBoard(input: {
  ctx: AdapterExecutionContext;
  hermesRunId: string;
  request: HermesApprovalRequest;
  dedupeKey: string;
  timeoutSec: number;
  signal: AbortSignal;
  redact: (value: string) => string;
  onKeepalive: (waitedMs: number) => Promise<void>;
}): Promise<ApprovalDecision> {
  const { ctx } = input;
  const issueId = issueIdFromContext(ctx);
  if (!issueId || !ctx.authToken) {
    return { choice: "deny", reason: "board_unavailable", detail: !issueId ? "run has no issue to ask on" : "run has no Paperclip auth token" };
  }
  const base = boardApiBase(ctx);
  const interactionsUrl = `${base}/api/issues/${encodeURIComponent(issueId)}/interactions`;
  let interactionId: string | null = null;
  try {
    const response = await fetch(interactionsUrl, {
      method: "POST",
      headers: boardHeaders(ctx),
      signal: input.signal,
      body: JSON.stringify({
        kind: "request_confirmation",
        resolverPolicy: "human_only",
        continuationPolicy: "none",
        idempotencyKey: `hermes-approval:${ctx.runId}:${input.dedupeKey}`.slice(0, 255),
        sourceRunId: ctx.runId,
        title: "Approve a flagged Hermes command?",
        summary: input.redact(input.request.description ?? input.request.command ?? "Hermes flagged a command").slice(0, 1000),
        payload: {
          version: 1,
          prompt: `Allow ${ctx.agent.name} to run this flagged command once? No answer within ${input.timeoutSec}s denies it.`.slice(0, 1000),
          acceptLabel: "Approve once",
          rejectLabel: "Deny",
          allowDeclineReason: false,
          detailsMarkdown: boardDetails(input),
        },
      }),
    });
    const body = asRecord(await response.json().catch(() => null));
    interactionId = nonEmpty(body?.id);
    if (!response.ok || !interactionId) {
      return { choice: "deny", reason: "board_unavailable", detail: `could not create board confirmation (HTTP ${response.status})` };
    }
  } catch (err) {
    return { choice: "deny", reason: "board_unavailable", detail: err instanceof Error ? err.message : String(err) };
  }

  const startedAt = Date.now();
  const deadline = startedAt + Math.max(1, input.timeoutSec) * 1000;
  let lastKeepalive = startedAt;
  while (!input.signal.aborted && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, BOARD_POLL_INTERVAL_MS));
    if (Date.now() - lastKeepalive >= APPROVAL_KEEPALIVE_MS) {
      lastKeepalive = Date.now();
      await input.onKeepalive(lastKeepalive - startedAt);
    }
    try {
      const response = await fetch(interactionsUrl, { method: "GET", headers: boardHeaders(ctx), signal: input.signal });
      if (!response.ok) continue;
      const list = await response.json().catch(() => null);
      const rows: BoardInteraction[] = Array.isArray(list) ? list : (asRecord(list)?.interactions as BoardInteraction[] | undefined) ?? [];
      const status = nonEmpty(rows.find((row) => row?.id === interactionId)?.status);
      if (status === "accepted") return { choice: "once", reason: "board_accepted", detail: `interaction ${interactionId}` };
      if (status && status !== "pending") return { choice: "deny", reason: "board_rejected", detail: `interaction ${interactionId} ${status}` };
    } catch {
      // A transient read failure keeps waiting until the deadline.
    }
  }

  await fetch(`${interactionsUrl}/${encodeURIComponent(interactionId)}/withdraw`, {
    method: "POST",
    headers: boardHeaders(ctx),
    body: JSON.stringify({ reason: `No board answer within ${input.timeoutSec}s; Hermes command denied.` }),
  }).catch(() => undefined);
  return { choice: "deny", reason: "board_timeout", detail: `no answer within ${input.timeoutSec}s (interaction ${interactionId})` };
}

export function approvalBoardTimeoutSec(value: unknown): number {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number.parseFloat(value) : Number.NaN;
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_APPROVAL_BOARD_TIMEOUT_SEC;
  return Math.min(3_600, Math.max(5, parsed));
}
