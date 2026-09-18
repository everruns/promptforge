# promptforge-model-client

This crate owns the Everruns-backed model transport and model-binding vocabulary.

- This is an Everruns vendor client (OpenAI completions driver, OpenRouter driver for OpenRouter hosts), not a universal transport. Other protocols use separate clients.
- The client does not depend on a parser, Lua runtime, store, observer, or executor. Executors adapt to it.
- Metrics vocabulary is canonical in `shared-promptforge-api`. This crate parses responses into those types and never defines a parallel metrics model.
- Hidden cross-crate seams let executors reach non-host internals. They must not gain documented status without a design change.
