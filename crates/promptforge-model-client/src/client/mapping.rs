//! Everruns mapping for the model transport: local messages, tools, and
//! options become Everruns chat values, vendor failures become
//! [`CompletionError`]s, and vendor streams accumulate into finished turns.
//! The mapping is deterministic given its inputs (the only clock read is the
//! turn timer started by the caller), so behavior tests cover it without a
//! vendor.

use std::num::NonZeroU32;
use std::time::Instant;

use everruns_provider::{
    AgentLoopError, ClientSideTool, DeferrablePolicy, LlmCallConfig, LlmMessage, LlmMessageRole,
    LlmStreamEvent, Provider, ReasoningEffort, ToolCall as VendorToolCall, ToolDefinition,
    ToolHints,
};
use futures_util::StreamExt;
use serde_json::Value;
use shared_promptforge_api::events::ClientTiming;

use super::{Message, StreamDelta, ToolCall, ToolSchema};
use crate::{
    Error, Result, Usage,
    model::{CompletionError, CompletionOptions},
};

/// Phrases for an empty turn, carried over from the previous transport.
pub(crate) const EMPTY_REPLY: &str = "empty model reply";
pub(crate) const EMPTY_REPLY_REASONING_IGNORED: &str =
    "empty model reply; reasoning output ignored";

/// Bounds an accumulated turn to the configured byte cap.
pub(crate) fn check_cap(len: usize, cap: u64) -> Result<()> {
    if len as u64 > cap {
        return Err(Error::MalformedResponse(format!(
            "response stream exceeds the {cap}-byte limit"
        )));
    }
    Ok(())
}

/// Escapes control characters in a diagnostic body and bounds it to `max` chars.
///
/// Control characters (including newlines and carriage returns) are rendered in
/// their `\u{..}`/`\n` escaped form so a backend body cannot forge log lines or
/// smuggle terminal control sequences into a diagnostic. An empty body is
/// reported as a fixed marker.
pub(crate) fn escape_controls(body: &str, max: usize) -> String {
    if body.is_empty() {
        return "(empty body)".to_owned();
    }
    let mut escaped = String::with_capacity(body.len());
    for ch in body.chars().take(max) {
        if ch.is_control() {
            for part in ch.escape_default() {
                escaped.push(part);
            }
        } else {
            escaped.push(ch);
        }
    }
    escaped
}

/// Maps local messages onto Everruns chat messages.
pub(crate) fn map_messages(
    messages: &[Message],
) -> std::result::Result<Vec<LlmMessage>, CompletionError> {
    messages.iter().map(map_message).collect()
}

/// Maps one local message onto an Everruns chat message.
///
/// Multimodal content (a parts array rather than a string) has no Everruns
/// equivalent at this layer, so it travels as its compact JSON rendering
/// instead of being dropped.
fn map_message(message: &Message) -> std::result::Result<LlmMessage, CompletionError> {
    let role = match message.role.as_str() {
        "system" => LlmMessageRole::System,
        "user" => LlmMessageRole::User,
        "assistant" => LlmMessageRole::Assistant,
        "tool" => LlmMessageRole::Tool,
        other => {
            return Err(CompletionError::from(Error::InvalidConfig(format!(
                "unsupported message role: {other}"
            ))));
        }
    };
    let text = match &message.content {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    let mut mapped = LlmMessage::text(role, text);
    let is_tool = matches!(mapped.role, LlmMessageRole::Tool);
    if let Some(calls) = &message.tool_calls {
        let mut mapped_calls = Vec::with_capacity(calls.len());
        for raw in calls {
            mapped_calls.push(map_assistant_tool_call(raw)?);
        }
        mapped.tool_calls = Some(mapped_calls);
    }
    if let Some(id) = &message.tool_call_id {
        mapped.tool_call_id = Some(id.clone());
    } else if is_tool {
        return Err(CompletionError::from(Error::InvalidConfig(
            "tool message is missing tool_call_id".into(),
        )));
    }
    Ok(mapped)
}

/// Maps one assistant tool-call JSON value onto a vendor tool call.
///
/// Two shapes arrive here: the flat local shape
/// (`{ "id", "name", "arguments" }`) and the OpenAI transcript shape the
/// executor stores (`{ "id", "type": "function", "function": {
/// "name", "arguments" } }`). The nested shape wins when present.
fn map_assistant_tool_call(raw: &Value) -> std::result::Result<VendorToolCall, CompletionError> {
    let function = raw.get("function").unwrap_or(raw);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| raw.get("name").and_then(Value::as_str))
        .ok_or_else(|| {
            CompletionError::from(Error::InvalidConfig(format!(
                "assistant tool call is missing its name: {}",
                escape_controls(&raw.to_string(), 500)
            )))
        })?;
    let arguments = function
        .get("arguments")
        .or_else(|| raw.get("arguments"))
        .cloned()
        .unwrap_or(Value::Null);
    Ok(VendorToolCall {
        id: raw
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        name: name.to_owned(),
        arguments,
    })
}

/// Maps local tool schemas onto Everruns client-side tools.
pub(crate) fn map_tools(
    tools: &[ToolSchema],
) -> std::result::Result<Vec<ToolDefinition>, CompletionError> {
    tools
        .iter()
        .map(|tool| {
            Ok(ToolDefinition::ClientSide(ClientSideTool {
                name: tool.name.clone(),
                display_name: None,
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                category: None,
                deferrable: DeferrablePolicy::default(),
                hints: ToolHints::default(),
                full_parameters: None,
            }))
        })
        .collect()
}

/// Maps a vendor failure onto the local error taxonomy.
///
/// The mapping preserves what downstream code matches on: failures that
/// carry an HTTP status keep it (plus the bounded, escaped vendor message)
/// so `status()`/`backend_body()` and every downstream match keep working;
/// anything else becomes a transport failure with the vendor message
/// preserved. Vendor messages embed their status as
/// `OpenAI API error ({status} ...)`, which is parsed back out so the exact
/// reported status survives.
pub(crate) fn map_llm_error(error: AgentLoopError) -> CompletionError {
    use everruns_provider::LlmErrorKind as Kind;
    if error.is_model_not_available() {
        let mut body = escape_controls(&error.to_string(), 2000);
        if let Some(id) = error.model_not_available_id() {
            body.push_str(&format!(" (model: {id})"));
        }
        return CompletionError::from(Error::Backend { status: 404, body });
    }
    if error.is_request_too_large() {
        return CompletionError::from(Error::Backend {
            status: 413,
            body: escape_controls(&error.to_string(), 2000),
        });
    }
    let body = escape_controls(&error.to_string(), 2000);
    if let Some(status) = sniff_vendor_status(&body) {
        return CompletionError::from(Error::Backend { status, body });
    }
    let lowered = body.to_lowercase();
    if lowered.contains("model not found")
        || lowered.contains("unknown model")
        || lowered.contains("model does not exist")
    {
        return CompletionError::from(Error::Backend { status: 404, body });
    }
    if lowered.contains("context length")
        || lowered.contains("context_length")
        || lowered.contains("maximum context")
    {
        return CompletionError::from(Error::Backend { status: 413, body });
    }
    let status = match error.llm_error_kind() {
        Some(Kind::Authentication) => 401,
        Some(Kind::RateLimited) => 429,
        Some(Kind::InvalidRequest) => 400,
        Some(Kind::Unavailable) => 503,
        Some(Kind::QuotaExhausted) | Some(Kind::BillingPressure { .. }) => 402,
        Some(Kind::AttestationRequired) => 403,
        Some(Kind::Other) | None => 502,
        Some(_) => 502,
    };
    CompletionError::from(Error::Backend { status, body })
}

/// Recovers the vendor-reported HTTP status from a driver message of the
/// form `OpenAI API error ({status} ...)`. Returns `None` when the message
/// carries no status.
pub(crate) fn sniff_vendor_status(body: &str) -> Option<u16> {
    let marker = "API error (";
    let start = body.find(marker)? + marker.len();
    let digits: String = body[start..]
        .chars()
        .take_while(|c: &char| c.is_numeric())
        .collect();
    if digits.len() == 3 {
        digits.parse().ok()
    } else {
        None
    }
}

/// The [`LlmCallConfig`] the client builds for an invocation.
pub(crate) fn build_call_config(
    model: &str,
    options: &CompletionOptions,
    tools: Vec<ToolDefinition>,
) -> LlmCallConfig {
    let mut config = LlmCallConfig::new(model);
    config.tools = tools;
    config.temperature = options
        .temperature
        .map(|temperature| temperature.get() as f32);
    config.max_tokens = options.max_tokens.map(NonZeroU32::get);
    config.reasoning_effort = options
        .thinking
        .and_then(|thinking| thinking.then_some(ReasoningEffort::Medium));
    config
}

/// A finished turn accumulated from the vendor stream.
pub(crate) struct FinishedTurn {
    pub(crate) text: String,
    pub(crate) reasoning: String,
    pub(crate) tool_calls: Vec<ToolCall>,
    pub(crate) finish_reason: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) usage: Usage,
    pub(crate) timing: ClientTiming,
}

/// Streams one turn to completion, reporting deltas as they arrive.
///
/// The terminal `Done` event carries the serving model name, finish reason,
/// and token usage; a stream that ends without it is malformed. Reasoning
/// arrives as delta strings and reasoning items, both joined into one text.
pub(crate) async fn accumulate_turn(
    provider: &Provider,
    messages: Vec<LlmMessage>,
    config: &LlmCallConfig,
    cap: u64,
    started: Instant,
    on_delta: impl Fn(StreamDelta),
) -> std::result::Result<FinishedTurn, CompletionError> {
    let mut stream = provider
        .chat_completion_stream(messages, config)
        .await
        .map_err(map_llm_error)?;
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut finish_reason: Option<String> = None;
    let mut model: Option<String> = None;
    let mut usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        cached_tokens: None,
        reasoning_tokens: None,
    };
    let mut first_delta: Option<Instant> = None;
    let mut last_delta: Option<Instant> = None;
    let mut delta_chunks: u32 = 0;
    let mut done = false;
    while let Some(event) = stream.next().await {
        let event = event.map_err(map_llm_error)?;
        let now = Instant::now();
        if first_delta.is_none() {
            first_delta = Some(now);
        }
        last_delta = Some(now);
        match event {
            LlmStreamEvent::TextDelta(delta) => {
                // Empty content deltas (usage/finish frames) carry no text;
                // they still mark arrival time above but never reach the
                // callback, matching the previous transport.
                if delta.is_empty() {
                    continue;
                }
                text.push_str(&delta);
                check_cap(text.len() + reasoning.len(), cap).map_err(CompletionError::from)?;
                delta_chunks += 1;
                on_delta(StreamDelta::Text(delta));
            }
            LlmStreamEvent::ReasoningDelta { delta, .. } => {
                // Reasoning deltas are live progress only: the terminal item
                // below carries the persistable text, and accumulating both
                // would double it.
                if delta.is_empty() {
                    continue;
                }
                delta_chunks += 1;
                on_delta(StreamDelta::Reasoning(delta));
            }
            // The terminal artifact: the driver re-emits the full reasoning
            // text as an item at Done, so items accumulate while deltas
            // only report. Items never reach the callback: their text was
            // already reported as deltas.
            LlmStreamEvent::ReasoningItem(part) => {
                if let Some(part_text) = part.display_text() {
                    reasoning.push_str(&part_text);
                    check_cap(text.len() + reasoning.len(), cap).map_err(CompletionError::from)?;
                }
            }
            LlmStreamEvent::ToolCalls(calls) => {
                tool_calls = calls
                    .into_iter()
                    .map(|call| ToolCall {
                        id: call.id,
                        name: call.name,
                        arguments: call.arguments,
                    })
                    .collect();
            }
            LlmStreamEvent::Done(metadata) => {
                done = true;
                finish_reason.clone_from(&metadata.finish_reason);
                model.clone_from(&metadata.model);
                usage = Usage {
                    prompt_tokens: metadata.prompt_tokens.unwrap_or(0),
                    completion_tokens: metadata.completion_tokens.unwrap_or(0),
                    total_tokens: metadata.total_tokens.unwrap_or(0),
                    cached_tokens: metadata.cache_read_tokens,
                    reasoning_tokens: None,
                };
            }
            LlmStreamEvent::Error(error) => {
                return Err(CompletionError::from(Error::Backend {
                    status: error.status.unwrap_or(502),
                    body: escape_controls(&error.to_string(), 2000),
                }));
            }
            _ => {}
        }
    }
    if !done {
        return Err(CompletionError::from(Error::MalformedResponse(
            "response stream ended before its [DONE] sentinel".into(),
        )));
    }
    let timing = ClientTiming {
        ttft_ms: first_delta.map(|at| duration_ms(at.duration_since(started))),
        mean_itl_ms: match (first_delta, last_delta) {
            (Some(first), Some(last)) if delta_chunks >= 2 => {
                Some(duration_ms(last.duration_since(first)) / f64::from(delta_chunks - 1))
            }
            _ => None,
        },
        e2e_ms: duration_ms(started.elapsed()),
    };
    Ok(FinishedTurn {
        text,
        reasoning,
        tool_calls,
        finish_reason,
        model,
        usage,
        timing,
    })
}

/// A duration as fractional milliseconds.
fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// A whole-turn timeout, boxed as a transport failure source.
#[derive(Debug)]
pub(crate) struct TimeoutElapsed {
    pub(crate) after: std::time::Duration,
}

impl std::fmt::Display for TimeoutElapsed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "request timed out after {:?}", self.after)
    }
}

impl std::error::Error for TimeoutElapsed {}
