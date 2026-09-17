//! Command-line runner for a promptforge prompt against a live gateway.
//!
//! `cargo run -p promptforge-api --example run-prompt -- <prompt.md> [args]`
//!
//! The gateway URL comes from `PROMPTFORGE_GATEWAY_URL` (the client's own
//! environment contract) and the model the run binds its declared roles to
//! from `PROMPTFORGE_MODEL`, a name the gateway's catalog serves.

use std::num::NonZeroU32;
use std::sync::Arc;

use promptforge_api::client::GatewayClient;
use promptforge_api::{CapabilityRegistry, Environment, Prompt, RunContext, RunResult, Web};
use shared_promptforge_api::models::{ModelDescriptor, ModelId, ThinkingMode};
use shared_promptforge_api::observe::{Observation, Observer};

/// Prints each lifecycle observation so the run is visible from the terminal.
struct Trace;

impl Observer for Trace {
    fn observe(&self, execution: &str, section: &str, event: Observation) {
        eprintln!("[{execution}/{section}] {event:?}");
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut command_line = std::env::args().skip(1);
    let path = command_line
        .next()
        .ok_or("usage: run-prompt <prompt.md> [args]")?;
    let args: String = command_line.collect::<Vec<_>>().join(" ");

    let model_name = std::env::var("PROMPTFORGE_MODEL")?;
    let context = NonZeroU32::new(
        std::env::var("PROMPTFORGE_MODEL_CONTEXT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(128_000),
    )
    .ok_or("context is non-zero")?;
    let model = ModelDescriptor::new(
        ModelId::gateway(model_name.clone())?,
        "The gateway model this run binds every declared role to",
        context,
        ThinkingMode::Never,
    );

    let source = std::fs::read_to_string(&path)?;
    let observer: Arc<dyn Observer> = Arc::new(Trace);
    let prompt = Prompt::parse(&source, &path, observer.as_ref())?;

    // The shipped web prompt declares `promptforge/web`; a capability the
    // host never registered resolves as absent and fails prepare, so the
    // runner installs it from the same gateway root the client uses.
    let base_url = std::env::var("PROMPTFORGE_GATEWAY_URL")?;
    let token = std::env::var("PROMPTFORGE_GATEWAY_API_KEY").unwrap_or_else(|_| "local".to_owned());
    let mut registry = CapabilityRegistry::new();
    registry.register(Arc::new(Web::new(&base_url, token.clone())?))?;
    // The Everruns-backed nested agent, reaching its model back through the
    // same gateway so the vendor credential never leaves it.
    registry.register(Arc::new(promptforge_agent::Agents::with_model(
        &base_url,
        token,
        &model_name,
    )?))?;

    let env = Environment::new()
        .client(GatewayClient::from_env()?)
        .registry(registry);
    let ctx = RunContext::new(&path)
        .observer(Arc::clone(&observer))
        .model(model);

    match env.run(&prompt, &args, ctx).await {
        RunResult::Ok(text) => {
            println!("{text}");
            Ok(())
        }
        other => Err(format!("run did not complete: {other:?}").into()),
    }
}
