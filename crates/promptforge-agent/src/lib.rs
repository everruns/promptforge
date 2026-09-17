//! The `promptforge/agent` capability: one nested Everruns agent turn.
//!
//! PromptForge's own model surface is a single completion: `models.infer`
//! sends one tool-free round and returns its text, and `models.loop` drives
//! the tool loop the executor owns. Neither carries an identity across
//! rounds. This capability contributes the other shape - a named agent with
//! its own instructions, its own iteration budget, and its own tool loop -
//! by delegating the turn to the Everruns framework, the same runtime
//! `svit` and `yolop` build on.
//!
//! The nested agent reaches its model through the PromptForge gateway,
//! whose OpenAI-compatible API root Everruns' OpenAI provider accepts as a
//! `base_url`. The vendor credential stays in the gateway: this crate holds
//! only the gateway root and the shared bearer, exactly as
//! `promptforge-web` does for search.
//!
//! # Examples
//! ```
//! use promptforge_agent::Agents;
//! use shared_promptforge_api::capabilities::Capability;
//!
//! let capability = Agents::new("https://gateway.example.com/v1", "bearer-token")?;
//! assert_eq!(capability.id().to_string(), "promptforge/agent");
//! # Ok::<(), shared_promptforge_api::tools::ToolError>(())
//! ```

use std::sync::Arc;

use shared_promptforge_api::capabilities::{
    Capability, CapabilityError, CapabilityErrorKind, CapabilityId, Contribution, RunServices,
};
use shared_promptforge_api::tools::{Tool, ToolError, ToolErrorKind, ToolId, ToolOutput};

/// The longest prompt the nested agent accepts, in bytes. A nested turn is
/// a delegation, not a transcript carrier: the caller summarizes first.
const MAX_PROMPT_BYTES: usize = 256 * 1024;

/// The longest instructions the caller may set on the nested agent.
const MAX_INSTRUCTIONS_BYTES: usize = 32 * 1024;

/// The nested agent's own tool-loop budget, independent of the calling
/// prompt's `max_tool_iterations`: a runaway nested loop must not be able
/// to spend the caller's budget.
const MAX_ITERATIONS: usize = 8;

/// How the nested agent reaches its model.
///
/// `Gateway` is the shipped shape: the PromptForge gateway's
/// OpenAI-compatible root plus its shared bearer, so the vendor credential
/// never leaves the gateway. `Simulated` is the offline shape: Everruns'
/// deterministic simulator answers with the configured text and needs no
/// credential or network, which is what the tests and the offline smoke
/// path use.
#[derive(Debug, Clone)]
enum Backing {
    /// The gateway root (for example `http://127.0.0.1:8099/v1`), its
    /// bearer token, and the catalog model id the nested agent runs on.
    Gateway {
        base_url: String,
        token: String,
        model: String,
    },
    /// The deterministic simulated reply.
    Simulated { response: String },
}

/// The first-party `promptforge/agent` capability.
///
/// Contributes `promptforge/agent/run`: one nested Everruns agent turn,
/// returning the agent's final text. Configuration is validated at
/// construction - at host startup - rather than at a run's prepare time.
#[derive(Debug, Clone)]
pub struct Agents {
    /// The stable identity, `promptforge/agent`.
    id: CapabilityId,
    /// The pre-built tool, cloned into each run's contribution.
    run: AgentRun,
}

impl Agents {
    /// Builds the capability against a gateway root and bearer token.
    ///
    /// The nested agent runs on the `default_model` catalog id unless a
    /// call names another.
    ///
    /// # Errors
    /// Returns a [`ToolError`] with [`ToolErrorKind::InvalidArguments`]
    /// when `base_url` is empty or not an `http`/`https` URL, or when
    /// `token` is empty.
    pub fn new(base_url: &str, token: impl Into<String>) -> Result<Agents, ToolError> {
        Self::with_model(base_url, token, "gpt-4o-mini")
    }

    /// Builds the capability naming the nested agent's default model.
    ///
    /// # Errors
    /// As [`Agents::new`], and additionally when `default_model` is empty.
    pub fn with_model(
        base_url: &str,
        token: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Result<Agents, ToolError> {
        let token = token.into();
        let model = default_model.into();
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(invalid(
                "promptforge/agent: base_url must be an http or https gateway API root",
            ));
        }
        if token.is_empty() {
            return Err(invalid("promptforge/agent: token must not be empty"));
        }
        if model.is_empty() {
            return Err(invalid("promptforge/agent: model must not be empty"));
        }
        Ok(Agents {
            id: CapabilityId::from_validated("promptforge/agent"),
            run: AgentRun {
                backing: Backing::Gateway {
                    base_url: base_url.to_owned(),
                    token,
                    model,
                },
            },
        })
    }

    /// Builds the capability over Everruns' deterministic simulator.
    ///
    /// Every nested turn answers with `response`, with no credential and no
    /// network, so a test or an offline smoke run exercises the same
    /// activation, dispatch, and result path as the gateway shape.
    #[must_use]
    pub fn simulated(response: impl Into<String>) -> Agents {
        Agents {
            id: CapabilityId::from_validated("promptforge/agent"),
            run: AgentRun {
                backing: Backing::Simulated {
                    response: response.into(),
                },
            },
        }
    }
}

/// Builds the invalid-arguments error every constructor check returns.
fn invalid(message: &str) -> ToolError {
    ToolError::message(message).with_kind(ToolErrorKind::InvalidArguments)
}

impl Capability for Agents {
    fn id(&self) -> &CapabilityId {
        &self.id
    }

    #[expect(
        clippy::unnecessary_literal_bound,
        reason = "the Capability trait fixes this return type to &str, so the &'static str suggestion cannot be applied"
    )]
    fn description(&self) -> &str {
        "Delegate a task to a nested agent that runs its own instructions and tool loop."
    }

    fn create(&self, services: &RunServices) -> Result<Contribution, CapabilityError> {
        if services.cancel.is_cancelled() {
            return Err(
                CapabilityError::message("promptforge/agent: the run was cancelled")
                    .with_kind(CapabilityErrorKind::Cancelled),
            );
        }
        Ok(Contribution {
            tools: vec![Arc::new(self.run.clone())],
        })
    }
}

/// The `promptforge/agent/run` tool: one nested Everruns agent turn.
#[derive(Debug, Clone)]
struct AgentRun {
    /// How this tool reaches its model.
    backing: Backing,
}

impl AgentRun {
    /// Runs one nested turn and returns the agent's final text.
    ///
    /// The Everruns `Agent` is built per call rather than once per host:
    /// `instructions` is a per-call argument, so the identity a call
    /// delegates to is the call's own. The engine is in-memory, so nothing
    /// survives the turn - a nested agent has no memory across
    /// `agent.run` calls, and the caller's prose is the only continuity.
    async fn run_once(&self, instructions: String, prompt: String) -> Result<String, ToolError> {
        let mut builder = everruns::Agent::builder()
            .name("promptforge-agent-run")
            .instructions(instructions)
            .max_iterations(MAX_ITERATIONS);
        builder = match &self.backing {
            Backing::Gateway {
                base_url,
                token,
                model,
            } => builder.model(model.clone()).provider(
                // Everruns' facade `OpenAI` provider speaks the Responses
                // API (`/v1/responses`); the PromptForge gateway serves
                // only `/v1/chat/completions`, so the nested agent takes
                // the driver crate's Chat Completions provider instead.
                everruns_openai::completions_provider("promptforge-gateway", token.clone())
                    .base_url(base_url.clone()),
            ),
            Backing::Simulated { response } => {
                builder.model(everruns::Model::simulated(response.clone()))
            }
        };
        let agent = builder.build().map_err(|error| {
            ToolError::message(format!("promptforge/agent: agent is not valid: {error}"))
                .with_kind(ToolErrorKind::InvalidArguments)
        })?;
        let turn = everruns::Engine::new()
            .create(agent)
            .send_and_wait(prompt)
            .await
            .map_err(|error| {
                ToolError::message(format!("promptforge/agent: nested turn failed: {error}"))
                    .with_kind(ToolErrorKind::Backend)
            })?;
        if turn.success {
            Ok(turn.response)
        } else {
            Err(
                ToolError::message("promptforge/agent: the nested agent did not complete its task")
                    .with_kind(ToolErrorKind::Backend),
            )
        }
    }
}

#[async_trait::async_trait]
impl Tool for AgentRun {
    fn id(&self) -> ToolId {
        ToolId::from_validated("promptforge/agent/run")
    }

    #[expect(
        clippy::unnecessary_literal_bound,
        reason = "the Tool trait fixes this return type to &str, so the &'static str suggestion cannot be applied"
    )]
    fn wire_name(&self) -> &str {
        "agent_run"
    }

    #[expect(
        clippy::unnecessary_literal_bound,
        reason = "the Tool trait fixes this return type to &str, so the &'static str suggestion cannot be applied"
    )]
    fn description(&self) -> &str {
        "Delegate one task to a nested agent and return the text it produced."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["prompt"],
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the nested agent.",
                    "minLength": 1,
                    "maxLength": MAX_PROMPT_BYTES
                },
                "instructions": {
                    "type": "string",
                    "description": "The nested agent's standing instructions.",
                    "maxLength": MAX_INSTRUCTIONS_BYTES
                }
            }
        })
    }

    async fn call(&self, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let prompt = args
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if prompt.is_empty() {
            return Err(invalid("promptforge/agent: prompt is required, got empty"));
        }
        if prompt.len() > MAX_PROMPT_BYTES {
            return Err(invalid(&format!(
                "promptforge/agent: prompt is {} bytes, over the {MAX_PROMPT_BYTES} byte limit",
                prompt.len()
            )));
        }
        let instructions = args
            .get("instructions")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Complete the supplied task.");
        if instructions.len() > MAX_INSTRUCTIONS_BYTES {
            return Err(invalid(&format!(
                "promptforge/agent: instructions are {} bytes, over the {MAX_INSTRUCTIONS_BYTES} byte limit",
                instructions.len()
            )));
        }
        // A nested agent's output is model text shaped by whatever it read
        // on the way, so it carries the same trust as any other model
        // reply reaching the caller: untrusted, and nonce-wrapped before
        // it can reach model input again.
        self.run_once(instructions.to_owned(), prompt.to_owned())
            .await
            .map(ToolOutput::untrusted)
    }
}

#[cfg(test)]
mod tests {
    use shared_promptforge_api::cancel::CancelHandle;
    use shared_promptforge_api::capabilities::{Capability, CapabilityErrorKind, RunServices};
    use shared_promptforge_api::tools::{ToolErrorKind, ToolId};

    use crate::Agents;

    /// Fresh run services over an empty VFS and a live cancel handle.
    fn services() -> RunServices {
        RunServices::new(shared_vfs::VfsRef::builder().build(), CancelHandle::new())
    }

    #[test]
    fn activating_contributes_the_run_tool_under_the_capability_id() {
        let capability = Agents::new("http://127.0.0.1:8099/v1", "tok").expect("valid config");
        let contribution = capability.create(&services()).expect("activation succeeds");
        let ids: Vec<ToolId> = contribution.tools.iter().map(|tool| tool.id()).collect();
        assert_eq!(
            ids,
            vec![ToolId::parse("promptforge/agent/run").expect("valid tool id")]
        );
    }

    #[test]
    fn a_cancelled_run_activates_nothing() {
        let capability = Agents::new("http://127.0.0.1:8099/v1", "tok").expect("valid config");
        let cancel = CancelHandle::new();
        cancel.cancel();
        let services = RunServices::new(shared_vfs::VfsRef::builder().build(), cancel);
        let error = capability
            .create(&services)
            .expect_err("a cancelled run must not activate");
        assert_eq!(error.kind(), CapabilityErrorKind::Cancelled);
    }

    #[test]
    fn a_non_http_base_url_is_refused_at_construction() {
        let error = Agents::new("ftp://gateway.example.com", "tok")
            .expect_err("a non-http root is refused");
        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }

    #[test]
    fn an_empty_token_is_refused_at_construction() {
        let error =
            Agents::new("http://127.0.0.1:8099/v1", "").expect_err("an empty token is refused");
        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }

    #[tokio::test]
    async fn the_simulated_backing_runs_a_nested_turn_with_no_credential() {
        let capability = Agents::simulated("nested answer");
        let contribution = capability.create(&services()).expect("activation succeeds");
        let tool = contribution.tools.first().expect("one contributed tool");

        let output = tool
            .call(serde_json::json!({"prompt": "do the thing"}))
            .await
            .expect("the simulated nested turn succeeds");

        assert_eq!(output.text(), "nested answer");
    }

    #[tokio::test]
    async fn an_empty_prompt_is_refused_before_any_turn() {
        let capability = Agents::simulated("unreachable");
        let contribution = capability.create(&services()).expect("activation succeeds");
        let tool = contribution.tools.first().expect("one contributed tool");

        let error = tool
            .call(serde_json::json!({"prompt": ""}))
            .await
            .expect_err("an empty prompt is refused");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
    }
}
