//! An Everruns-backed chat completions client, pointed at one vendor endpoint.
//!
//! The client drives the Everruns OpenAI-compatible completions driver and
//! always streams: every call accumulates the deltas into one [`Completion`]
//! - a text reply or the tool calls the model asked for - while invoking the
//! caller's delta callback with each live [`StreamDelta`]. A caller with no
//! use for deltas passes a no-op closure. [`GatewayClient::complete`] sends
//! a `tools` array when the caller supplies one, so the executor's
//! tool-call loop runs over this client. The client holds only the vendor's
//! base URL and, when one is set, the bearer key. Point the endpoint at
//! OpenAI or at an OpenAI-compatible base such as OpenRouter to retarget
//! it; a loopback mock vendor needs no key.

mod config;
pub(crate) mod mapping;
mod transport;
mod wire;

pub use config::{
    GatewayEndpoint, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_DEFAULT_BASE_URL, OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL, OPENROUTER_DEFAULT_BASE_URL, SecretError, SecretString,
};
// Canonical in `shared-promptforge-api`; re-exported so the
// `promptforge_model_client::client::StreamDelta` path keeps resolving.
pub use shared_promptforge_api::wire::StreamDelta;
pub use transport::{GatewayClient, RequestLimits};
#[doc(hidden)]
pub use wire::ToolSchemaError;
pub use wire::{Completion, CompletionResult, Message, ToolArguments, ToolCall, ToolSchema};

#[cfg(test)]
mod tests;
