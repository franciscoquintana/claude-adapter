// Model-gated debug logging for nvidia/moonshotai/kimi-k2.6.
// When the model matches, every event is appended as one JSON line to
// ~/.claude-adapter/debug_kimi/<YYYY-MM-DD>.jsonl. Other models are no-ops.

import { join } from 'path';
import { getBaseDir, ensureDirExists, appendJsonLine, getTodayDateString } from './fileStorage';

const KIMI_MODEL_ID = 'nvidia/moonshotai/kimi-k2.6';
const DEBUG_DIR = join(getBaseDir(), 'debug_kimi');
const MAX_PAYLOAD_BYTES = 10_240;

function todayFile(): string {
    return join(DEBUG_DIR, `${getTodayDateString()}.jsonl`);
}

function truncate(value: unknown): unknown {
    if (typeof value !== 'string') return value;
    if (value.length <= MAX_PAYLOAD_BYTES) return value;
    return value.slice(0, MAX_PAYLOAD_BYTES) + `…[truncated ${value.length - MAX_PAYLOAD_BYTES}b]`;
}

function sanitize(payload: unknown): unknown {
    if (payload === null || payload === undefined) return payload;
    if (typeof payload === 'string') return truncate(payload);
    if (Array.isArray(payload)) return payload.map(sanitize);
    if (typeof payload === 'object') {
        const out: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(payload as Record<string, unknown>)) {
            out[k] = sanitize(v);
        }
        return out;
    }
    return payload;
}

export function isKimiModel(model: string | undefined): boolean {
    return model === KIMI_MODEL_ID;
}

export function kimiDebug(
    model: string | undefined,
    event: string,
    payload: unknown,
    requestId?: string,
): void {
    if (!isKimiModel(model)) return;
    try {
        ensureDirExists(DEBUG_DIR);
        appendJsonLine(todayFile(), {
            ts: new Date().toISOString(),
            requestId: requestId ?? null,
            event,
            payload: sanitize(payload),
        });
    } catch {
        // never let the debug logger crash a request
    }
}
