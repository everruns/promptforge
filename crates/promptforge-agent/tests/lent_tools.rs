//! The nested agent's own tool loop reaches a lent PromptForge tool.
//!
//! The claim this file exists to keep honest is that
//! [`Agents::with_tool`](promptforge_agent::Agents::with_tool) is a real
//! bridge and not just a builder that compiles: the nested Everruns agent
//! must actually call the PromptForge tool during its own loop, without
//! the calling prompt orchestrating the round trip.
//!
//! The test needs a live gateway, because the deterministic simulator
//! answers without ever calling a tool. With `PROMPTFORGE_GATEWAY_URL`
//! unset it returns early rather than failing, so an offline `cargo test`
//! stays green; set that variable (and `PROMPTFORGE_MODEL`) to run it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use promptforge_agent::Agents;
use shared_promptforge_api::cancel::CancelHandle;
use shared_promptforge_api::capabilities::{Capability, RunServices};
use shared_promptforge_api::tools::{Tool, ToolError, ToolId, ToolOutput};

/// A tool whose output only the tool itself can supply: the nested agent
/// cannot produce the sentinel from its own knowledge, so the sentinel
/// appearing in the final text proves the call happened.
#[derive(Debug)]
struct Sentinel {
    called: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl Tool for Sentinel {
    fn id(&self) -> ToolId {
        ToolId::from_validated("promptforge/agent/sentinel")
    }

    #[expect(
        clippy::unnecessary_literal_bound,
        reason = "the Tool trait fixes this return type to &str"
    )]
    fn wire_name(&self) -> &str {
        "lookup_launch_code"
    }

    #[expect(
        clippy::unnecessary_literal_bound,
        reason = "the Tool trait fixes this return type to &str"
    )]
    fn description(&self) -> &str {
        "Return today's launch code. The only way to learn the launch code."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        self.called.store(true, Ordering::SeqCst);
        Ok(ToolOutput::trusted("XYZZY-42"))
    }
}

#[tokio::test]
async fn the_nested_agent_calls_a_lent_tool_in_its_own_loop() {
    let Ok(base_url) = std::env::var("PROMPTFORGE_GATEWAY_URL") else {
        // No gateway configured: nothing to prove against, and a missing
        // environment is not a failure.
        return;
    };
    let token = std::env::var("PROMPTFORGE_GATEWAY_API_KEY").unwrap_or_else(|_| "local".to_owned());
    let model = std::env::var("PROMPTFORGE_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_owned());

    let called = Arc::new(AtomicBool::new(false));
    let capability = Agents::with_model(&base_url, token, model)
        .expect("valid configuration")
        .with_tool(Arc::new(Sentinel {
            called: Arc::clone(&called),
        }));
    let services = RunServices::new(shared_vfs::VfsRef::builder().build(), CancelHandle::new());
    let contribution = capability.create(&services).expect("activation succeeds");
    let tool = contribution.tools.first().expect("one contributed tool");

    let output = tool
        .call(serde_json::json!({
            "prompt": "What is today's launch code? Answer with the code only.",
            "instructions": "You have a tool that returns the launch code. Call it, then reply with the code only."
        }))
        .await
        .expect("the nested turn succeeds");

    assert!(
        called.load(Ordering::SeqCst),
        "the nested agent must call the lent tool in its own loop"
    );
    assert!(
        output.text().contains("XYZZY-42"),
        "the nested agent must report what the lent tool returned, got: {}",
        output.text()
    );
}
