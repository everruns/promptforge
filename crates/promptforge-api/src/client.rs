//! An Everruns-backed chat completions client, pointed at one vendor endpoint.
//!
//! The client drives the Everruns OpenAI-compatible completions driver
//! (or the OpenRouter driver for OpenRouter endpoints) and always streams
//! internally: [`GatewayClient::complete`] accumulates the deltas into one
//! text reply or the tool calls the model asked for, invoking the caller's
//! delta callback with each live delta. [`GatewayClient::complete`] sends a
//! `tools` array when the caller supplies one, so the executor's tool-call
//! loop runs over this client. The client holds only the vendor's base URL
//! and the bearer key, selected from the environment ([`OPENAI_API_KEY`] /
//! [`OPENROUTER_API_KEY`] with [`OPENAI_BASE_URL`]/[`OPENROUTER_BASE_URL`]
//! overrides). Point a base URL at a local OpenAI-compatible server to
//! retarget it. [`fetch_model_catalog`] reads the vendor's model list for
//! host-side concerns (the Workshop dropdown and its selection resolution);
//! the list never crosses into the environment an executor run prepares
//! against.
//!
//! The implementation lives in the `promptforge-model-client` crate and is
//! re-exported here: hosts pass a [`GatewayClient`] to
//! [`RunContext::client`](crate::RunContext) and classify its failures through
//! [`CompletionError`].

pub use promptforge_model_client::client::{
    GatewayClient, GatewayEndpoint, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_DEFAULT_BASE_URL,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_DEFAULT_BASE_URL, RequestLimits,
    SecretString,
};
pub use promptforge_model_client::model::{
    CompletionError, CompletionErrorKind, fetch_model_catalog,
};

pub(crate) use promptforge_model_client::client::{
    Completion, CompletionResult, Message, StreamDelta, ToolCall, ToolSchema,
};
