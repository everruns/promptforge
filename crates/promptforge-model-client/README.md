# promptforge-model-client

The PromptForge model client and model-catalog vocabulary: the
Everruns-backed chat-completions transport plus the catalog and
prompt-local binding vocabulary.

## Layout

- `client::GatewayClient`: builds from a `GatewayEndpoint` plus a vendor
  bearer key (or from the environment) and runs one chat completion to a
  full `Completion`, streaming progress into the caller's delta callback.
  The backend is Everruns — the OpenAI-compatible completions driver for
  OpenAI endpoints (including OpenAI-compatible mocks) and the OpenRouter
  driver for OpenRouter endpoints — so there is no gateway hop and the
  vendor credential stays with the caller.
- `client::{Message, ToolSchema, Completion, StreamDelta}`: the wire types
  the client exchanges. Assistant tool calls arrive in either the flat
  local shape or the OpenAI transcript shape the executor stores.
- `client::{OPENAI_API_KEY, OPENROUTER_API_KEY, ...}`: environment
  selection. `OPENAI_API_KEY` selects OpenAI (overridable via
  `OPENAI_BASE_URL`); otherwise `OPENROUTER_API_KEY` selects OpenRouter
  (overridable via `OPENROUTER_BASE_URL`). A loopback base URL may omit
  the key for mock vendors.
- `model::fetch_model_catalog`: lists the vendor's models (driver listing
  first, direct `GET {base}/models` for hosts the driver cannot list) and
  shapes them into a `ModelCatalog`. Entries without a context window are
  skipped.
- `model::{ModelId, ModelBinding, ModelSet, ModelView}`: the validated
  model identity plus the prompt-local types a host resolves and freezes
  model selections through.

## Errors

Vendor failures keep their HTTP status with the bounded, escaped vendor
message (`Backend`); anything else becomes `Transport`. The taxonomy —
`Disabled`, `Transport`, `Backend`, `MalformedResponse`, `EmptyReply`,
`MissingConfiguration`, `InvalidConfiguration` — is unchanged from the
gateway era, so downstream matches keep working.

The crate contains no prompt parser, no Lua runtime, and no executor; it is
the Everruns-backed model client only, never a universal client.
