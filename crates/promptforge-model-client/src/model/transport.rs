//! Model catalog sourced from the bound Everruns vendor.
//!
//! [`fetch_model_catalog`] lists the vendor's models through the same
//! Everruns driver the chat transport uses and shapes them into a
//! [`ModelCatalog`]. When the driver cannot list a host (custom or mock
//! base URLs outside the driver's known hosts), the fetch falls back to a
//! direct `GET {base}/models` in the previous gateway list shape, so local
//! OpenAI-compatible servers keep working. The `(base_url, token)` shape
//! stays: an empty token sends no bearer credential (the vendor decides), a
//! set token travels as one.
//!
//! Like the previous transport, entries without a context window are skipped:
//! a catalog entry must be an inference model with a known window.

use std::num::NonZeroU32;
use std::sync::LazyLock;

use serde::Deserialize;
use shared_promptforge_api::models::{ModelCatalog, ModelDescriptor, ModelId, ThinkingMode};

use super::super::client::{GatewayClient, GatewayEndpoint, SecretString};
use crate::Error;
use crate::client::mapping::{escape_controls, map_llm_error};
use crate::model::CompletionError;

/// Host recorded on catalog entries when the endpoint URL has none.
const UNKNOWN_SERVER: &str = "unknown";

/// Cap on a catalog response body: the largest plausible registry payload.
const MAX_CATALOG_BODY_BYTES: u64 = 16 * 1024 * 1024;

/// Bound on an error body kept for diagnostics.
const MAX_ERROR_BODY_CHARS: usize = 2000;

/// Shared HTTP client for the direct-listing fallback path.
static CATALOG_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(reqwest::Client::new);

/// One entry of the previous gateway list shape.
#[derive(Debug, Deserialize)]
struct CatalogModelEntry {
    /// Vendor model id, for example `"qwen3-30b"`.
    id: String,
    /// Human-readable label; empty when the vendor sends none.
    #[serde(default)]
    description: String,
    /// Advertised context window; entries without one are skipped below.
    context: Option<u32>,
    /// Advertised thinking support, if the vendor reports any.
    thinking: Option<ThinkingMode>,
}

/// The previous gateway list shape.
#[derive(Debug, Deserialize)]
struct ModelsListResponse {
    data: Vec<CatalogModelEntry>,
}

/// Fetches the vendor's model catalog: driver listing first, direct `GET
/// {base}/models` when the driver cannot list the host.
///
/// A vendor failure maps through the same taxonomy as
/// [`GatewayClient::complete`].
///
/// [`GatewayClient::complete`]: crate::client::GatewayClient::complete
pub async fn fetch_model_catalog(
    base_url: &str,
    token: &str,
) -> std::result::Result<ModelCatalog, CompletionError> {
    if let Some(catalog) = fetch_via_driver(base_url, token).await? {
        return Ok(catalog);
    }
    fetch_via_listing(base_url, token).await
}

/// Lists models through the Everruns driver, or `None` when the driver
/// cannot list the host (custom or mock base URLs).
fn fetch_via_driver(
    base_url: &str,
    token: &str,
) -> impl std::future::Future<Output = std::result::Result<Option<ModelCatalog>, CompletionError>> {
    async move {
        let endpoint = GatewayEndpoint::new(base_url).map_err(CompletionError::from)?;
        let base = endpoint.url.clone();
        let client = if token.is_empty() {
            GatewayClient::keyless(endpoint)
        } else {
            let key = SecretString::new(token).map_err(|error| {
                CompletionError::from(Error::InvalidConfig(format!("invalid token: {error}")))
            })?;
            GatewayClient::new(endpoint, key)
        };
        let provider = client.provider().map_err(CompletionError::from)?;
        let discovered = provider.list_models().await.map_err(map_llm_error)?;
        let Some(discovered) = discovered else {
            return Ok(None);
        };
        let server = catalog_server(&base);
        let mut descriptors = Vec::new();
        for model in &discovered {
            let Some(context) = model
                .discovered_profile
                .as_ref()
                .and_then(|profile| profile.limits.as_ref())
                .map(|limits| limits.context)
                .and_then(|context| u32::try_from(context).ok())
                .and_then(NonZeroU32::new)
            else {
                continue;
            };
            let id = ModelId::new(&server, &model.model_id).map_err(|error| {
                CompletionError::from(Error::InvalidConfig(format!(
                    "vendor model id is not a valid model id: {error}"
                )))
            })?;
            descriptors.push(ModelDescriptor::new(
                id,
                model.display_name.clone().unwrap_or(model.model_id.clone()),
                context,
                ThinkingMode::Never,
            ));
        }
        let catalog = ModelCatalog::new(descriptors).map_err(|error| {
            CompletionError::from(Error::InvalidConfig(format!(
                "vendor model catalog is not a valid catalog: {error}"
            )))
        })?;
        Ok(Some(catalog))
    }
}

/// Lists models with a direct `GET {base}/models` in the previous gateway
/// list shape: the fallback for hosts the driver cannot list.
async fn fetch_via_listing(
    base_url: &str,
    token: &str,
) -> std::result::Result<ModelCatalog, CompletionError> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let mut request = CATALOG_CLIENT.get(&url);
    if !token.is_empty() {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.map_err(Error::http)?;
    let status = response.status();
    if !status.is_success() {
        let body = read_error_body(response).await?;
        return Err(CompletionError::from(Error::Backend {
            status: status.as_u16(),
            body,
        }));
    }
    let body = read_catalog_body(response).await?;
    let parsed: ModelsListResponse = serde_json::from_slice(&body).map_err(|error| {
        CompletionError::from(Error::MalformedResponseSource {
            message: "model catalog response is not valid JSON".to_owned(),
            source: error.into(),
        })
    })?;
    let mut descriptors = Vec::with_capacity(parsed.data.len());
    for entry in parsed.data {
        let Some(context) = entry.context.and_then(NonZeroU32::new) else {
            continue;
        };
        let thinking = entry.thinking.ok_or_else(|| {
            CompletionError::from(Error::MalformedResponse(format!(
                "model entry {:?} has a context window but no thinking mode",
                entry.id
            )))
        })?;
        let id = ModelId::gateway(&entry.id).map_err(|error| {
            CompletionError::from(Error::MalformedResponse(format!(
                "model entry {:?} is not a valid model id: {error}",
                entry.id
            )))
        })?;
        descriptors.push(ModelDescriptor::new(
            id,
            entry.description,
            context,
            thinking,
        ));
    }
    ModelCatalog::new(descriptors).map_err(|error| {
        CompletionError::from(Error::InvalidConfig(format!(
            "vendor model catalog is not a valid catalog: {error}"
        )))
    })
}

/// Reads a catalog body through the size cap.
async fn read_catalog_body(
    response: reqwest::Response,
) -> std::result::Result<Vec<u8>, CompletionError> {
    use futures_util::StreamExt as _;
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(Error::http)?;
        if body.len() as u64 + chunk.len() as u64 > MAX_CATALOG_BODY_BYTES {
            return Err(CompletionError::from(Error::MalformedResponse(format!(
                "model catalog response exceeds the {MAX_CATALOG_BODY_BYTES}-byte limit"
            ))));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Reads an error body for diagnostics, bounded and escaped.
async fn read_error_body(
    response: reqwest::Response,
) -> std::result::Result<String, CompletionError> {
    let status = response.status();
    let body = response.text().await.map_err(|error| {
        CompletionError::from(Error::BackendBodyRead {
            status: status.as_u16(),
            source: Box::new(error),
        })
    })?;
    Ok(escape_controls(&body, MAX_ERROR_BODY_CHARS))
}

/// The server label recorded on catalog entries: the endpoint host.
fn catalog_server(base_url: &str) -> String {
    url::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| {
            if base_url.is_empty() {
                UNKNOWN_SERVER.into()
            } else {
                base_url.into()
            }
        })
}
