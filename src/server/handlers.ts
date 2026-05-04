// Proxy server request handlers
import { FastifyRequest, FastifyReply } from 'fastify';
import OpenAI from 'openai';
import { Agent, fetch as undiciFetch } from 'undici';
import { AnthropicMessageRequest } from '../types/anthropic';
import { AdapterConfig } from '../types/config';
import { convertRequestToOpenAI } from '../converters/request';
import { convertResponseToAnthropic, createErrorResponse } from '../converters/response';
import { streamOpenAIToAnthropic } from '../converters/streaming';
import { streamXmlOpenAIToAnthropic } from '../converters/xmlStreaming';
import { validateAnthropicRequest, formatValidationErrors } from '../utils/validation';
import { logger, RequestLogger } from '../utils/logger';
import { recordUsage } from '../utils/tokenUsage';
import { recordError } from '../utils/errorLog';
import { kimiDebug } from '../utils/kimiDebug';

// Request ID counter for unique identification
let requestIdCounter = 0;

function generateRequestId(): string {
    requestIdCounter++;
    const timestamp = Date.now().toString(36);
    const counter = requestIdCounter.toString(36).padStart(4, '0');
    return `req_${timestamp}_${counter} `;
}

// undici dispatcher with disabled body timeout: models like minimax-m2.7 pause
// >30s between chunks during reasoning, which would otherwise abort the stream.
const longLivedDispatcher = new Agent({
    bodyTimeout: 0,
    headersTimeout: 600_000,
    keepAliveTimeout: 10_000,
    keepAliveMaxTimeout: 600_000,
});

// Use undici's fetch directly so our Agent dispatcher actually applies.
// Node's global fetch ships its own bundled undici and would ignore an
// external Agent from a different version, causing instant connection errors.
const longLivedFetch = ((input: any, init?: any) =>
    undiciFetch(input, { ...(init ?? {}), dispatcher: longLivedDispatcher })) as unknown as typeof fetch;

/**
 * Handle POST /v1/messages requests
 */
export function createMessagesHandler(config: AdapterConfig) {
    const openai = new OpenAI({
        baseURL: config.baseUrl,
        apiKey: config.apiKey,
        timeout: 600_000,
        fetch: longLivedFetch,
    });

    return async (request: FastifyRequest, reply: FastifyReply): Promise<void> => {
        const requestId = generateRequestId();
        const log = logger.withRequestId(requestId);

        // Add request ID to response headers for client tracing
        reply.header('X-Request-Id', requestId);

        try {
            // Validate request before processing
            const validation = validateAnthropicRequest(request.body);
            if (!validation.valid) {
                const errorMessage = formatValidationErrors(validation.errors);
                log.warn('Invalid request', { errors: validation.errors });
                const errorResponse = createErrorResponse(new Error(errorMessage), 400);
                reply.code(400).send({ error: errorResponse.error });
                return;
            }

            const anthropicRequest = request.body as AnthropicMessageRequest;
            const targetModel = anthropicRequest.model;
            const isStreaming = anthropicRequest.stream ?? false;

            log.info(`→ ${targetModel} [sent]`);

            kimiDebug(targetModel, 'anthropic_request', {
                model: targetModel,
                streaming: isStreaming,
                messageCount: anthropicRequest.messages?.length ?? 0,
                toolCount: anthropicRequest.tools?.length ?? 0,
                toolNames: anthropicRequest.tools?.map((t: any) => t.name) ?? [],
                lastMessage: summarizeMessage(anthropicRequest.messages?.[anthropicRequest.messages.length - 1]),
            }, requestId);

            // Determine tool calling style from config
            const toolStyle = config.toolFormat || 'native';

            // Convert request to OpenAI format
            const openaiRequest = convertRequestToOpenAI(anthropicRequest, targetModel, toolStyle);

            kimiDebug(targetModel, 'openai_request', {
                model: openaiRequest.model,
                messageCount: Array.isArray(openaiRequest.messages) ? openaiRequest.messages.length : 0,
                roles: Array.isArray(openaiRequest.messages) ? openaiRequest.messages.map((m: any) => m.role) : [],
                toolCount: Array.isArray((openaiRequest as any).tools) ? (openaiRequest as any).tools.length : 0,
                tools: Array.isArray((openaiRequest as any).tools)
                    ? (openaiRequest as any).tools.map((t: any) => ({
                          name: t.function?.name,
                          description: typeof t.function?.description === 'string' ? t.function.description.slice(0, 200) : undefined,
                      }))
                    : [],
            }, requestId);

            // Log tool calling mode when tools are present
            if (toolStyle === 'xml' && anthropicRequest.tools?.length) {
                log.info(`Using XML tool calling mode (${anthropicRequest.tools.length} tools)`);
            }

            if (isStreaming) {
                if (toolStyle === 'xml') {
                    await handleXmlStreamingRequest(openai, openaiRequest, reply, anthropicRequest.model, config.baseUrl, log, requestId);
                } else {
                    await handleStreamingRequest(openai, openaiRequest, reply, anthropicRequest.model, config.baseUrl, log, requestId);
                }
            } else {
                await handleNonStreamingRequest(openai, openaiRequest, reply, anthropicRequest.model, config.baseUrl, log, requestId);
            }

            log.info(`← ${targetModel} [received]`);
        } catch (error) {
            const body = request.body as any;
            handleError(error as Error, reply, log, {
                requestId,
                provider: config.baseUrl,
                modelName: body?.model ?? 'unknown',
                streaming: body?.stream ?? false
            });
        }
    };
}

/**
 * Handle non-streaming API request
 */
async function handleNonStreamingRequest(
    openai: OpenAI,
    openaiRequest: any,
    reply: FastifyReply,
    originalModel: string,
    provider: string,
    log: RequestLogger,
    requestId?: string
): Promise<void> {
    log.debug('Making non-streaming request');

    const response = await openai.chat.completions.create({
        ...openaiRequest,
        stream: false,
    });

    log.debug('Response received', {
        finishReason: response.choices[0]?.finish_reason,
        usage: response.usage
    });

    const choice = response.choices[0];
    const message = choice?.message as any;
    const toolCalls = Array.isArray(message?.tool_calls) ? message.tool_calls : [];
    kimiDebug(originalModel, 'nonstream_response', {
        finishReason: choice?.finish_reason ?? null,
        hasContent: typeof message?.content === 'string' && message.content.length > 0,
        contentPreview: typeof message?.content === 'string' ? message.content.slice(0, 500) : null,
        toolCallCount: toolCalls.length,
        toolCalls: toolCalls.map((tc: any) => {
            const args = tc.function?.arguments ?? '';
            let jsonValid = false;
            let parseError: string | undefined;
            try {
                JSON.parse(args);
                jsonValid = true;
            } catch (e) {
                parseError = e instanceof Error ? e.message : String(e);
            }
            return {
                id: tc.id ?? null,
                idLooksOpenAi: typeof tc.id === 'string' && tc.id.startsWith('call_'),
                name: tc.function?.name,
                argsLength: typeof args === 'string' ? args.length : 0,
                args,
                jsonValid,
                parseError,
            };
        }),
        usage: response.usage,
    }, requestId);

    // Record token usage
    if (response.usage) {
        recordUsage({
            provider,
            modelName: originalModel,
            model: response.model,
            inputTokens: response.usage.prompt_tokens,
            outputTokens: response.usage.completion_tokens,
            cachedInputTokens: response.usage.prompt_tokens_details?.cached_tokens,
            streaming: false
        });
    }

    const anthropicResponse = convertResponseToAnthropic(response as any, originalModel);
    reply.send(anthropicResponse);
}

/**
 * Handle streaming API request
 */
async function handleStreamingRequest(
    openai: OpenAI,
    openaiRequest: any,
    reply: FastifyReply,
    originalModel: string,
    provider: string,
    log: RequestLogger,
    requestId?: string
): Promise<void> {
    log.debug('Making streaming request');

    const stream = await openai.chat.completions.create({
        ...openaiRequest,
        stream: true,
    } as OpenAI.ChatCompletionCreateParamsStreaming);

    const estimatedInputTokens = estimateInputTokens(openaiRequest.messages);
    await streamOpenAIToAnthropic(stream as any, reply, originalModel, provider, estimatedInputTokens, requestId);
    log.debug('Streaming completed');
}

/**
 * Handle XML streaming API request (for models without native tool calling)
 */
async function handleXmlStreamingRequest(
    openai: OpenAI,
    openaiRequest: any,
    reply: FastifyReply,
    originalModel: string,
    provider: string,
    log: RequestLogger,
    requestId?: string
): Promise<void> {
    log.debug('Making XML streaming request (experimental)');

    const stream = await openai.chat.completions.create({
        ...openaiRequest,
        stream: true,
    } as OpenAI.ChatCompletionCreateParamsStreaming);

    const estimatedInputTokens = estimateInputTokens(openaiRequest.messages);
    await streamXmlOpenAIToAnthropic(stream as any, reply, originalModel, provider, estimatedInputTokens, requestId);
    log.debug('XML streaming completed');
}

function summarizeMessage(msg: unknown): unknown {
    if (!msg || typeof msg !== 'object') return null;
    const m = msg as any;
    const blockTypes = Array.isArray(m.content)
        ? m.content.map((b: any) => (typeof b === 'string' ? 'text' : b?.type ?? 'unknown'))
        : typeof m.content === 'string'
            ? ['text']
            : [];
    return {
        role: m.role,
        contentTypes: blockTypes,
        toolUseIds: Array.isArray(m.content)
            ? m.content.filter((b: any) => b?.type === 'tool_use').map((b: any) => b.id)
            : [],
        toolResultIds: Array.isArray(m.content)
            ? m.content.filter((b: any) => b?.type === 'tool_result').map((b: any) => b.tool_use_id)
            : [],
    };
}

/**
 * Rough character-based input token estimate. Used to populate `input_tokens`
 * in the initial `message_start` event for upstreams (e.g. NVIDIA glm-5.1) that
 * only deliver real `usage` info in the final stream chunk. The real value
 * overwrites this estimate as soon as upstream usage arrives.
 */
function estimateInputTokens(messages: unknown): number {
    if (!Array.isArray(messages)) return 0;
    let chars = 0;
    for (const msg of messages) {
        if (!msg || typeof msg !== 'object') continue;
        const content = (msg as any).content;
        if (typeof content === 'string') {
            chars += content.length;
        } else if (Array.isArray(content)) {
            for (const part of content) {
                if (typeof part === 'string') chars += part.length;
                else if (part && typeof part === 'object' && typeof (part as any).text === 'string') {
                    chars += (part as any).text.length;
                }
            }
        }
        const toolCalls = (msg as any).tool_calls;
        if (Array.isArray(toolCalls)) {
            for (const tc of toolCalls) {
                const name = tc?.function?.name ?? '';
                const args = tc?.function?.arguments ?? '';
                chars += String(name).length + String(args).length;
            }
        }
    }
    return Math.ceil(chars / 4);
}

/**
 * Handle errors and send appropriate response
 */
function handleError(
    error: Error,
    reply: FastifyReply,
    log: RequestLogger,
    context?: { requestId: string; provider: string; modelName: string; streaming: boolean }
): void {
    let statusCode = 500;

    // Try to extract status code from OpenAI error
    if ('status' in error) {
        statusCode = (error as any).status;
    }

    log.error('Request failed', error, { statusCode });

    // Record error to file if context is available
    if (context) {
        recordError(error, context);
    }

    const errorResponse = createErrorResponse(error, statusCode);
    reply.code(errorResponse.status).send({ error: errorResponse.error });
}


