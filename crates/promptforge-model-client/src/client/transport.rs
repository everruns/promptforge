//! [`GatewayClient`]: the Everruns-backed chat transport.
//!
//! The client keeps its established shape: callers build it from a
//! [`GatewayEndpoint`] plus a bearer key (or from the environment), then call
//! [`GatewayClient::complete`] with the conversation, the tool schemas, and
//! the invocation options. What changed is the backend: instead of POSTing an
//! OpenAI-shaped request to a gateway over HTTP, the client drives the
//! Everruns OpenAI-compatible completions driver
//! ([`OpenAICompletionsChatDriver`]) directly. The driver speaks the same
//! `/chat/completions` wire format (including server-sent-event streaming),
//! so the observable contract - message roles, tool calls, usage reporting,
//! and the whole [`CompletionError`] taxonomy - is unchanged.
//!
//! The endpoint selects the vendor driver. OpenAI endpoints (including
//! OpenAI-compatible loopback mocks) use the Everruns OpenAI-compatible
//! completions driver; OpenRouter endpoints use the Everruns OpenRouter
//! driver, which carries OpenRouter's listing and headers. Both speak the
//! same provider contract, so the mapping is identical. There is exactly one
//! transport: no gateway-versus-Everruns switch, no provider enum in this
//! crate's API.
//!
//! # Examples
//!
//! ```no_run
//! use promptforge_model_client::client::{GatewayClient, GatewayEndpoint, Message, SecretString};
//! use promptforge_model_client::model::CompletionOptions;
//!
//! # async fn run() -> Result<(), promptforge_model_client::model::CompletionError> {
//! let endpoint = GatewayEndpoint::new("https://api.openai.com/v1")?;
//! let client = GatewayClient::new(endpoint, SecretString::new("sk-test-key").expect("key"));
//! let options = CompletionOptions::new("gpt-4o-mini");
//! let completion = client
//!     .complete(&[Message::user("hi")], None, &options, |_delta| {})
//!     .await?;
//! let _ = completion.result();
//! # Ok(())
//! # }
//! ```
//!
//! # Reachable [`CompletionError`] kinds
//!
//! - F1 `Disabled`: the client was built by [`GatewayClient::disabled`].
//! - F2 `Transport`: the vendor call timed out or its stream failed before a
//!   complete turn arrived.
//! - F3 `Backend`: the vendor rejected the call (authentication, unknown
//!   model, rate limit, bad request, outage). The status and the bounded,
//!   escaped vendor message travel in the error for the log tail.
//! - F4 `MalformedResponse`: the accumulated turn overflowed the byte cap, a
//!   tool-call batch was truncated by `length`/`content_filter` (partial
//!   arguments must not execute), or the stream ended before its `[DONE]`
//!   sentinel.
//! - F5 `EmptyReply`: the turn carried neither text nor tool calls.
//! - F6 `MissingConfiguration`: no vendor key was configured (see
//!   [`GatewayClient::from_env`]).
//! - F7 `InvalidConfiguration`: a caller-side value cannot be sent (unknown
//!   message role, tool message without `tool_call_id`, unparseable
//!   assistant tool call).
//!
//! [`CompletionError`]: crate::model::CompletionError
//! [`OpenAICompletionsChatDriver`]: everruns_openai::OpenAICompletionsChatDriver

use std::num::{NonZeroU32, NonZeroU64};
use std::time::{Duration, Instant};

use everruns_openai::OpenAICompletionsChatDriver;
use everruns_openrouter::OpenRouterChatDriver;
use everruns_provider::{BearerAuth, Provider};
use serde_json::{Value, json};
use tracing::debug;

use super::{
    Completion, CompletionResult, GatewayEndpoint, Message, SecretString, StreamDelta, ToolSchema,
    mapping::{
        EMPTY_REPLY, EMPTY_REPLY_REASONING_IGNORED, TimeoutElapsed, accumulate_turn,
        build_call_config, map_messages, map_tools,
    },
};
// Re-exported at its previous path for the test suite, which escapes diagnostics.
#[cfg(test)]
pub(crate) use super::mapping::escape_controls;
use crate::{
    Error, Result,
    model::{CompletionError, CompletionOptions},
};

/// Default request timeout: the whole turn, including streaming, must finish
/// within it.
pub(crate) const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Default response cap, in bytes of accumulated text.
pub(crate) const DEFAULT_MAX_RESPONSE_BYTES: u64 = 4 * 1024 * 1024;

/// Builds the Everruns provider for a base URL: the OpenRouter driver for
/// OpenRouter hosts, the OpenAI-compatible completions driver otherwise
/// (OpenAI itself and OpenAI-compatible mocks share the wire shape).
fn driver_for(base_url: &str) -> Provider {
    if is_openrouter_url(base_url) {
        Provider::new("promptforge", OpenRouterChatDriver::new())
    } else {
        Provider::new("promptforge", OpenAICompletionsChatDriver::new())
    }
}

/// Whether a base URL points at OpenRouter.
fn is_openrouter_url(base_url: &str) -> bool {
    url::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.contains("openrouter")))
        .unwrap_or(false)
}

/// Run limits applied to one client.
#[derive(Debug, Clone, Copy)]
pub struct RequestLimits {
    /// Whole-turn timeout, including streaming.
    pub timeout: Duration,
    /// Cap on accumulated response text, in bytes.
    pub max_response_bytes: u64,
}

/// The single model transport: an Everruns chat driver bound to one vendor
/// endpoint plus its bearer key.
///
/// The `GatewayClient`/`GatewayEndpoint` names stay so every call site keeps
/// working; only the backend changed (previously an HTTP gateway, now
/// Everruns). `base_url`/`key` are kept for the [`GatewayClient::endpoint`]
/// accessor and redacted debug output.
#[derive(Clone)]
#[non_exhaustive]
pub struct GatewayClient {
    provider: Option<Provider>,
    base_url: String,
    key: Option<SecretString>,
    request_timeout: Duration,
    max_response_bytes: u64,
}

impl GatewayClient {
    /// Build a client that sends the endpoint's bearer key to the endpoint.
    ///
    /// The driver is the Everruns OpenAI-compatible completions driver; an
    /// endpoint whose URL is an OpenAI-compatible base (OpenAI itself, or
    /// OpenRouter's `https://openrouter.ai/api/v1`) selects the vendor. The
    /// key travels as a bearer credential and is never logged.
    #[must_use]
    pub fn new(endpoint: GatewayEndpoint, key: SecretString) -> GatewayClient {
        let base_url = endpoint.url.clone();
        let provider = driver_for(&base_url)
            .base_url(base_url.clone())
            .auth(BearerAuth::new(key.expose().to_owned()));
        GatewayClient {
            provider: Some(provider),
            base_url: endpoint.url,
            key: Some(key),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }

    /// Build a client that presents no bearer key.
    ///
    /// No `Authorization` header goes out. This fits a loopback mock vendor,
    /// which trusts keyless callers; against any other vendor the calls fail
    /// with a `Backend` authentication error. Nothing here checks the
    /// endpoint's host - the caller decides, and [`GatewayClient::from_env`]
    /// decides by [`GatewayEndpoint::is_loopback`].
    ///
    /// # Examples
    ///
    /// ```
    /// use promptforge_model_client::client::{GatewayClient, GatewayEndpoint};
    ///
    /// let endpoint = GatewayEndpoint::new("http://127.0.0.1:8081/v1")?;
    /// let client = GatewayClient::keyless(endpoint);
    /// let _ = client;
    /// # Ok::<(), promptforge_model_client::model::CompletionError>(())
    /// ```
    #[must_use]
    pub fn keyless(endpoint: GatewayEndpoint) -> GatewayClient {
        let base_url = endpoint.url.clone();
        let provider = driver_for(&base_url).base_url(base_url.clone());
        GatewayClient {
            provider: Some(provider),
            base_url: endpoint.url,
            key: None,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }

    /// Build a client that cannot read vendor configuration or send requests.
    ///
    /// Hosts use this explicit sentinel for hermetic execution paths. Any
    /// attempted model call fails with a `Disabled`-kind [`CompletionError`].
    ///
    /// # Examples
    ///
    /// ```
    /// # async fn run() {
    /// use promptforge_model_client::client::{GatewayClient, Message};
    /// use promptforge_model_client::model::{CompletionErrorKind, CompletionOptions};
    ///
    /// let client = GatewayClient::disabled();
    /// let options = CompletionOptions::new("m");
    /// let error = client
    ///     .complete(&[Message::user("hi")], None, &options, |_delta| {})
    ///     .await
    ///     .expect_err("a disabled client cannot complete");
    /// assert_eq!(error.kind(), CompletionErrorKind::Disabled);
    /// # }
    /// ```
    #[must_use]
    pub fn disabled() -> GatewayClient {
        GatewayClient {
            provider: None,
            base_url: String::new(),
            key: None,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }

    /// The vendor base URL this client talks to (empty for [`GatewayClient::disabled`]).
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.base_url
    }

    /// The bound Everruns provider, unless [`GatewayClient::disabled`].
    pub(crate) fn provider(&self) -> Result<&Provider> {
        self.provider.as_ref().ok_or(Error::GatewayDisabled)
    }

    /// Whether this client presents a bearer key; a test seam for the
    /// environment constructor, which never exposes the key itself.
    #[cfg(test)]
    pub(crate) fn has_key(&self) -> bool {
        self.key.is_some()
    }

    /// Applies the run's request limits to this client.
    ///
    /// Each completion (including its stream) is bounded by
    /// `request_timeout`, and accumulated response text is refused once it
    /// would exceed `max_response_bytes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroU64;
    /// use std::time::Duration;
    ///
    /// use promptforge_model_client::client::GatewayClient;
    ///
    /// let cap = NonZeroU64::new(1024 * 1024).ok_or("cap is non-zero")?;
    /// let client = GatewayClient::disabled().with_request_limits(Duration::from_secs(30), cap);
    /// let _ = client;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn with_request_limits(
        mut self,
        request_timeout: Duration,
        max_response_bytes: NonZeroU64,
    ) -> GatewayClient {
        self.request_timeout = request_timeout;
        self.max_response_bytes = max_response_bytes.get();
        self
    }

    /// Builds a client from the environment.
    ///
    /// [`OPENAI_API_KEY`] selects OpenAI ([`OPENAI_DEFAULT_BASE_URL`],
    /// overridable via [`OPENAI_BASE_URL`]); otherwise
    /// [`OPENROUTER_API_KEY`] selects OpenRouter
    /// ([`OPENROUTER_DEFAULT_BASE_URL`], overridable via
    /// [`OPENROUTER_BASE_URL`]). A loopback base URL may omit the key
    /// (keyless mock vendors); anywhere else a missing key is a
    /// `MissingConfiguration` error naming required versus actual.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use promptforge_model_client::client::GatewayClient;
    ///
    /// # fn run() -> Result<(), promptforge_model_client::model::CompletionError> {
    /// let client = GatewayClient::from_env()?;
    /// let _ = client;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_env() -> std::result::Result<GatewayClient, CompletionError> {
        super::config::from_env_with(|name| match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(Error::InvalidEnv(format!("{name} is not valid unicode")))
            }
        })
        .map_err(CompletionError::from)
    }

    /// Runs one chat completion to a full [`Completion`], streaming progress
    /// into `on_delta`.
    ///
    /// The driver always streams; text and reasoning deltas are reported as
    /// they arrive and the returned completion carries the accumulated turn.
    /// A caller with no use for deltas passes a no-op closure.
    pub async fn complete(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        options: &CompletionOptions,
        on_delta: impl Fn(StreamDelta),
    ) -> std::result::Result<Completion, CompletionError> {
        let provider = self.provider().map_err(CompletionError::from)?;
        let started = Instant::now();
        let ever_messages = map_messages(messages)?;
        let tools = tools.unwrap_or_default();
        let config = build_call_config(options.model.as_str(), options, map_tools(tools)?);
        let request_body = json!({
            "vendor": "everruns",
            "driver": "openai-completions",
            "endpoint": self.base_url,
            "model": options.model.as_str(),
            "messages": messages
                .iter()
                .map(|message| json!({"role": message.role, "content": message.content}))
                .collect::<Vec<_>>(),
            "tools": tools.len(),
            "stream": true,
            "temperature": options.temperature.map(|temperature| temperature.get()),
            "max_tokens": options.max_tokens.map(NonZeroU32::get),
            "thinking": options.thinking,
        });
        debug!(
            model = options.model.as_str(),
            messages = ever_messages.len(),
            tools = tools.len(),
            "everruns completion started"
        );

        let turn = tokio::time::timeout(
            self.request_timeout,
            accumulate_turn(
                provider,
                ever_messages,
                &config,
                self.max_response_bytes,
                started,
                on_delta,
            ),
        )
        .await
        .map_err(|_| {
            CompletionError::from(Error::Http(Box::new(TimeoutElapsed {
                after: self.request_timeout,
            })))
        })??;

        if !turn.tool_calls.is_empty()
            && matches!(
                turn.finish_reason.as_deref(),
                Some("length" | "content_filter")
            )
        {
            let reason = turn.finish_reason.clone().unwrap_or_default();
            return Err(CompletionError::from(Error::MalformedResponse(format!(
                "tool-call batch truncated by finish_reason {reason:?}: \
                 partial arguments must not execute"
            ))));
        }

        let result = if turn.tool_calls.is_empty() {
            if turn.text.trim().is_empty() {
                let detail = if turn.reasoning.trim().is_empty() {
                    EMPTY_REPLY
                } else {
                    EMPTY_REPLY_REASONING_IGNORED
                };
                return Err(CompletionError::from(Error::EmptyModelReply {
                    detail,
                    finish_reason: turn.finish_reason.clone(),
                }));
            }
            CompletionResult::Text(turn.text)
        } else {
            CompletionResult::ToolCalls(turn.tool_calls)
        };
        let reasoning_content = if turn.reasoning.trim().is_empty() {
            None
        } else {
            Some(turn.reasoning)
        };
        // A clean stop carries no finish reason downstream; anything else
        // (length, content_filter, tool-calls finish) survives.
        let finish_reason = turn
            .finish_reason
            .filter(|reason| reason.as_str() != "stop");
        let response_body = json!({
            "text": match &result {
                CompletionResult::Text(text) => Value::String(text.clone()),
                CompletionResult::ToolCalls(_) => Value::Null,
            },
            "tool_calls": match &result {
                CompletionResult::ToolCalls(calls) => json!(calls
                    .iter()
                    .map(|call| json!({
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }))
                    .collect::<Vec<_>>()),
                CompletionResult::Text(_) => Value::Null,
            },
            "finish_reason": finish_reason,
            "model": turn.model.clone(),
            "usage": {
                "prompt_tokens": turn.usage.prompt_tokens,
                "completion_tokens": turn.usage.completion_tokens,
                "total_tokens": turn.usage.total_tokens,
            },
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": match &result {
                        CompletionResult::Text(text) => Value::String(text.clone()),
                        CompletionResult::ToolCalls(_) => Value::Null,
                    },
                },
                "finish_reason": finish_reason,
            }],
        });
        Ok(Completion {
            result,
            finish_reason,
            reasoning_content,
            model: turn
                .model
                .unwrap_or_else(|| options.model.as_str().to_owned()),
            usage: Some(turn.usage),
            llama_timings: None,
            vllm_metrics: None,
            client_timing: Some(turn.timing),
            request_body,
            response_body,
        })
    }
}

impl std::fmt::Debug for GatewayClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayClient")
            .field("base_url", &self.base_url)
            .field(
                "key",
                &self
                    .key
                    .as_ref()
                    .map(|_| "<redacted>")
                    .unwrap_or("<redacted>"),
            )
            .field("request_timeout", &self.request_timeout)
            .field("max_response_bytes", &self.max_response_bytes)
            .finish_non_exhaustive()
    }
}

/// Builds a client from explicit environment values (a seam for tests).
///
/// Defined in [`config`](super::config) beside endpoint construction;
/// re-exported here so existing import paths keep working.
pub(crate) use super::config::from_env_with;
