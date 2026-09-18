//! The PromptForge model client and model-catalog vocabulary.
//!
//! [`client`] holds the Everruns-backed chat-completions transport:
//! [`client::GatewayClient`] drives the Everruns OpenAI-compatible
//! completions driver against one vendor base URL with a bearer key, and the
//! wire types ([`client::Message`], [`client::ToolSchema`],
//! [`client::Completion`], [`client::StreamDelta`]) are what it exchanges.
//! [`model`] holds the catalog and prompt-local binding vocabulary:
//! [`model::ModelCatalog`] built from the vendor's model listing, the
//! validated [`model::ModelId`] identity, and the
//! [`model::ModelBinding`]/[`model::ModelSet`]/[`model::ModelView`] types a
//! host resolves and freezes model selections through.
//!
//! The metrics vocabulary ([`Usage`], [`LlamaTimings`], [`VllmMetrics`],
//! [`ClientTiming`], [`CallMetrics`]) is canonical in
//! `shared-promptforge-api` and re-exported here: the client parses each
//! response body's call metadata into it, and [`client::Completion`] carries
//! the result. The model identity/catalog vocabulary ([`model::ModelId`],
//! [`model::ModelCatalog`], [`model::ModelDescriptor`],
//! [`model::ThinkingMode`]) and the streaming [`client::StreamDelta`] are
//! canonical there too and re-exported through their historical paths.
//!
//! The crate contains no prompt parser, no Lua runtime, and no executor; it is
//! the Everruns-backed model client only, never a universal client.

pub mod client;
mod error;
pub mod model;

#[doc(hidden)]
pub use crate::error::Error;
pub(crate) use crate::error::Result;

pub use shared_promptforge_api::events::{
    CallMetrics, ClientTiming, LlamaTimings, Usage, VllmMetrics,
};
