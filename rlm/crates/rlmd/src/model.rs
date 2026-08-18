use std::{net::IpAddr, sync::Arc, time::Duration};

use async_trait::async_trait;
use reqwest::{StatusCode, redirect::Policy};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    sync::Semaphore,
    time::{Instant, timeout_at},
};
use url::{Host, Url};

/// Hard ceiling for any response accepted from the model server.
///
/// MLX responses are normally only a few KiB. Keeping a ceiling here prevents a
/// broken local service from consuming the control plane's memory with an
/// unbounded response.
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const MAX_ERROR_SNIPPET_BYTES: usize = 4 * 1024;
pub const LOCAL_MODEL_ALIAS: &str = "default_model";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelReply {
    pub content: String,
    pub usage: ModelUsage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("invalid MLX base URL: {0}")]
    InvalidBaseUrl(String),

    #[error("invalid model configuration: {0}")]
    InvalidConfiguration(String),

    #[error("failed to build the local MLX HTTP client: {0}")]
    ClientBuild(#[source] reqwest::Error),

    #[error("the model generation queue was closed")]
    QueueClosed,

    #[error("local MLX request timed out after {0:?}")]
    Timeout(Duration),

    #[error("local MLX request failed: {0}")]
    Transport(#[source] reqwest::Error),

    #[error("local MLX returned HTTP {status}: {body}")]
    HttpStatus { status: StatusCode, body: String },

    #[error("local MLX response exceeded the {limit_bytes}-byte limit")]
    ResponseTooLarge { limit_bytes: usize },

    #[error("malformed local MLX response: {0}")]
    MalformedResponse(String),
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// Generate one non-streaming completion.
    ///
    /// There is deliberately no caller-supplied model name here. A deployment
    /// configures one fixed local model, so an API request cannot make MLX load a
    /// different model (or a remote Hugging Face model) by changing `model`.
    async fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: u32,
    ) -> Result<ModelReply, ModelError>;

    /// Verify that the local HTTP service is alive without running inference.
    async fn health(&self) -> Result<(), ModelError>;

    /// Verify that the chat-completions-compatible endpoint is ready without
    /// spending model tokens.
    async fn readiness(&self) -> Result<(), ModelError>;
}

/// Hardened client for a loopback-only `mlx_lm.server` instance.
///
/// The client never uses an environment proxy, never follows redirects, never
/// retries, and only accepts a literal loopback address or `localhost`. The
/// semaphore is held for the complete response body, serializing generations on
/// a memory-constrained inference host while allowing unrelated control-plane
/// work to remain concurrent.
#[derive(Clone)]
pub struct MlxModelClient {
    client: reqwest::Client,
    base_url: Url,
    model: String,
    request_timeout: Duration,
    generation_slots: Arc<Semaphore>,
}

impl std::fmt::Debug for MlxModelClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MlxModelClient")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("request_timeout", &self.request_timeout)
            .finish_non_exhaustive()
    }
}

impl MlxModelClient {
    pub fn new(base_url: Url, request_timeout: Duration) -> Result<Self, ModelError> {
        Self::with_semaphore(base_url, request_timeout, Arc::new(Semaphore::new(1)))
    }

    pub fn with_semaphore(
        base_url: Url,
        request_timeout: Duration,
        generation_slots: Arc<Semaphore>,
    ) -> Result<Self, ModelError> {
        validate_loopback_base_url(&base_url)?;

        if request_timeout.is_zero() {
            return Err(ModelError::InvalidConfiguration(
                "request timeout must be greater than zero".to_owned(),
            ));
        }

        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(Policy::none())
            .timeout(request_timeout)
            .build()
            .map_err(ModelError::ClientBuild)?;

        Ok(Self {
            client,
            base_url: normalized_base_url(base_url),
            model: LOCAL_MODEL_ALIAS.to_owned(),
            request_timeout,
            generation_slots,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    fn endpoint(&self, path: &str) -> Url {
        let mut endpoint = self.base_url.clone();
        endpoint.set_path(path);
        endpoint.set_query(None);
        endpoint.set_fragment(None);
        endpoint
    }

    async fn send_bounded(
        &self,
        request: reqwest::RequestBuilder,
        deadline: Instant,
    ) -> Result<(StatusCode, Vec<u8>), ModelError> {
        let operation = async {
            let response = request.send().await.map_err(ModelError::Transport)?;
            let status = response.status();
            let body = read_bounded_body(response).await?;
            Ok((status, body))
        };

        timeout_at(deadline, operation)
            .await
            .map_err(|_| ModelError::Timeout(self.request_timeout))?
    }

    fn require_success(status: StatusCode, body: &[u8]) -> Result<(), ModelError> {
        if status.is_success() {
            return Ok(());
        }

        Err(ModelError::HttpStatus {
            status,
            body: error_snippet(body),
        })
    }
}

#[async_trait]
impl ModelBackend for MlxModelClient {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: u32,
    ) -> Result<ModelReply, ModelError> {
        if messages.is_empty() {
            return Err(ModelError::InvalidConfiguration(
                "at least one chat message is required".to_owned(),
            ));
        }
        if max_tokens == 0 {
            return Err(ModelError::InvalidConfiguration(
                "max_tokens must be greater than zero".to_owned(),
            ));
        }

        let deadline = Instant::now() + self.request_timeout;
        let permit = timeout_at(deadline, self.generation_slots.clone().acquire_owned())
            .await
            .map_err(|_| ModelError::Timeout(self.request_timeout))?
            .map_err(|_| ModelError::QueueClosed)?;

        let payload = ChatCompletionRequest {
            model: &self.model,
            messages,
            max_tokens,
            stream: false,
        };
        let request = self
            .client
            .post(self.endpoint("/v1/chat/completions"))
            .json(&payload);

        // The owned permit is intentionally kept in this future until the body
        // is read. Dropping/cancelling the future drops the permit immediately.
        let result = self.send_bounded(request, deadline).await;
        // mlx_lm.server has no cancellation/status endpoint for an individual
        // generation. After a client-side timeout the accelerator job may still be
        // running, so allowing another request could overlap two generations
        // and exhaust host memory. Fail closed until rlmd is restarted.
        let ambiguous_in_flight = matches!(
            &result,
            Err(ModelError::Timeout(_) | ModelError::ResponseTooLarge { .. })
        ) || matches!(&result, Err(ModelError::Transport(error)) if !error.is_connect());
        if ambiguous_in_flight {
            self.generation_slots.close();
        }
        drop(permit);

        let (status, body) = result?;
        Self::require_success(status, &body)?;
        parse_chat_completion(&body)
    }

    async fn health(&self) -> Result<(), ModelError> {
        let deadline = Instant::now() + self.request_timeout;
        let request = self.client.get(self.endpoint("/health"));
        let (status, body) = self.send_bounded(request, deadline).await?;
        Self::require_success(status, &body)
    }

    async fn readiness(&self) -> Result<(), ModelError> {
        if self.generation_slots.is_closed() {
            return Err(ModelError::QueueClosed);
        }
        let deadline = Instant::now() + self.request_timeout;
        let request = self.client.get(self.endpoint("/v1/models"));
        let (status, body) = self.send_bounded(request, deadline).await?;
        Self::require_success(status, &body)?;

        let response: ModelsResponse = serde_json::from_slice(&body)
            .map_err(|error| ModelError::MalformedResponse(error.to_string()))?;
        if response.object != "list" {
            return Err(ModelError::MalformedResponse(
                "`/v1/models` response object was not `list`".to_owned(),
            ));
        }

        // Parsing at least one complete model entry proves this is the expected
        // API rather than an arbitrary HTTP 200. MLX may canonicalize local paths
        // in this listing, so exact string equality with the configured request
        // model would incorrectly reject valid relative paths.
        if !response
            .data
            .iter()
            .any(|model| !model.id.trim().is_empty())
        {
            return Err(ModelError::MalformedResponse(
                "`/v1/models` did not contain a usable model".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    max_tokens: u32,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
    usage: ModelUsage,
}

#[derive(Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatCompletionMessage {
    // The local backend may omit this key when generation immediately returns
    // EOS. The schema permits null content for an assistant tool-call message;
    // v0 does not act on tool calls, so both cases safely become an empty reply.
    content: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

fn parse_chat_completion(body: &[u8]) -> Result<ModelReply, ModelError> {
    let response: ChatCompletionResponse = serde_json::from_slice(body)
        .map_err(|error| ModelError::MalformedResponse(error.to_string()))?;
    let choice =
        response.choices.into_iter().next().ok_or_else(|| {
            ModelError::MalformedResponse("response contained no choices".to_owned())
        })?;

    Ok(ModelReply {
        content: choice.message.content.unwrap_or_default(),
        usage: response.usage,
        finish_reason: choice.finish_reason,
    })
}

async fn read_bounded_body(mut response: reqwest::Response) -> Result<Vec<u8>, ModelError> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        return Err(ModelError::ResponseTooLarge {
            limit_bytes: MAX_RESPONSE_BYTES,
        });
    }

    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(ModelError::Transport)? {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(ModelError::ResponseTooLarge {
                limit_bytes: MAX_RESPONSE_BYTES,
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn error_snippet(body: &[u8]) -> String {
    let end = body.len().min(MAX_ERROR_SNIPPET_BYTES);
    let mut snippet = String::from_utf8_lossy(&body[..end]).into_owned();
    if body.len() > end {
        snippet.push_str("...");
    }
    snippet
}

fn normalized_base_url(mut base_url: Url) -> Url {
    base_url.set_path("/");
    base_url.set_query(None);
    base_url.set_fragment(None);
    base_url
}

fn validate_loopback_base_url(base_url: &Url) -> Result<(), ModelError> {
    if !matches!(base_url.scheme(), "http" | "https") {
        return Err(ModelError::InvalidBaseUrl(
            "scheme must be http or https".to_owned(),
        ));
    }
    if !base_url.username().is_empty() || base_url.password().is_some() {
        return Err(ModelError::InvalidBaseUrl(
            "embedded credentials are not allowed".to_owned(),
        ));
    }
    if base_url.query().is_some() || base_url.fragment().is_some() {
        return Err(ModelError::InvalidBaseUrl(
            "query strings and fragments are not allowed".to_owned(),
        ));
    }

    let is_loopback = match base_url.host() {
        Some(Host::Ipv4(address)) => IpAddr::V4(address).is_loopback(),
        Some(Host::Ipv6(address)) => IpAddr::V6(address).is_loopback(),
        Some(Host::Domain(_)) => false,
        None => false,
    };
    if !is_loopback {
        return Err(ModelError::InvalidBaseUrl(
            "host must be a literal loopback IP address".to_owned(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{
        Json, Router,
        extract::State,
        routing::{get, post},
    };
    use serde_json::{Value, json};
    use tokio::{net::TcpListener, sync::Mutex, task::JoinHandle};

    use super::*;

    #[derive(Clone, Default)]
    struct Capture {
        request: Arc<Mutex<Option<Value>>>,
    }

    async fn spawn_server(router: Router) -> (Url, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (Url::parse(&format!("http://{address}")).unwrap(), task)
    }

    fn success_response(content: &str) -> Value {
        json!({
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18
            }
        })
    }

    #[test]
    fn accepts_only_http_loopback_base_urls() {
        for accepted in [
            "http://127.0.0.1:8080",
            "http://127.255.10.20:8080",
            "https://[::1]:8443",
        ] {
            let client = MlxModelClient::new(Url::parse(accepted).unwrap(), Duration::from_secs(1));
            assert!(client.is_ok(), "expected {accepted} to be accepted");
        }

        for rejected in [
            "http://example.com:8080",
            "https://192.168.1.10:8080",
            "http://0.0.0.0:8080",
            "http://localhost:8080",
            "ftp://127.0.0.1:8080",
            "http://user:password@localhost:8080",
            "http://localhost:8080?next=http://example.com",
        ] {
            let client = MlxModelClient::new(Url::parse(rejected).unwrap(), Duration::from_secs(1));
            assert!(client.is_err(), "expected {rejected} to be rejected");
        }
    }

    #[tokio::test]
    async fn sends_the_fixed_model_and_parses_content_and_usage() {
        async fn handler(
            State(capture): State<Capture>,
            Json(request): Json<Value>,
        ) -> Json<Value> {
            *capture.request.lock().await = Some(request);
            Json(success_response("local answer"))
        }

        let capture = Capture::default();
        let router = Router::new()
            .route("/v1/chat/completions", post(handler))
            .with_state(capture.clone());
        let (base_url, server) = spawn_server(router).await;
        let client = MlxModelClient::new(base_url, Duration::from_secs(2)).unwrap();

        let reply = client
            .chat(&[ChatMessage::new("user", "hello")], 32)
            .await
            .unwrap();

        assert_eq!(reply.content, "local answer");
        assert_eq!(reply.finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            reply.usage,
            ModelUsage {
                prompt_tokens: 11,
                completion_tokens: 7,
                total_tokens: 18,
            }
        );

        let request = capture.request.lock().await.clone().unwrap();
        assert_eq!(request["model"], LOCAL_MODEL_ALIAS);
        assert_eq!(request["stream"], false);
        assert_eq!(request["max_tokens"], 32);
        assert_eq!(request["messages"][0]["content"], "hello");

        server.abort();
    }

    #[tokio::test]
    async fn rejects_a_malformed_completion_response() {
        async fn malformed() -> Json<Value> {
            Json(json!({
                "choices": [],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 0,
                    "total_tokens": 1
                }
            }))
        }

        let router = Router::new().route("/v1/chat/completions", post(malformed));
        let (base_url, server) = spawn_server(router).await;
        let client = MlxModelClient::new(base_url, Duration::from_secs(2)).unwrap();

        let error = client
            .chat(&[ChatMessage::new("user", "hello")], 1)
            .await
            .unwrap_err();
        assert!(matches!(error, ModelError::MalformedResponse(_)));

        server.abort();
    }

    #[tokio::test]
    async fn health_and_readiness_do_not_run_inference() {
        let router = Router::new()
            .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            .route(
                "/v1/models",
                get(|| async {
                    Json(json!({
                        "object": "list",
                        "data": [{"id": "fixed-model", "object": "model"}]
                    }))
                }),
            );
        let (base_url, server) = spawn_server(router).await;
        let client = MlxModelClient::new(base_url, Duration::from_secs(2)).unwrap();

        client.health().await.unwrap();
        client.readiness().await.unwrap();

        server.abort();
    }

    #[tokio::test]
    async fn ambiguous_connection_loss_quarantines_generation_queue() {
        use tokio::io::AsyncReadExt;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buffer = [0_u8; 4096];
            let _ = stream.read(&mut buffer).await.unwrap();
            // Drop the connection after accepting the generation request but
            // before an HTTP response, leaving server-side completion state
            // ambiguous from the client's point of view.
        });
        let client = MlxModelClient::new(
            Url::parse(&format!("http://{address}")).unwrap(),
            Duration::from_secs(2),
        )
        .unwrap();
        let error = client
            .chat(&[ChatMessage::new("user", "hello")], 1)
            .await
            .unwrap_err();
        assert!(matches!(error, ModelError::Transport(_)));
        assert!(matches!(
            client.readiness().await,
            Err(ModelError::QueueClosed)
        ));
        server.await.unwrap();
    }
}
