use std::{net::IpAddr, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, StatusCode, redirect::Policy};
use rlm_protocol::{WorkerRequest, WorkerResponse};
use thiserror::Error;
use url::{Host, Url};

const MAX_WORKER_RESPONSE_BYTES: usize = 256 * 1024;

#[async_trait]
pub trait WorkerBackend: Send + Sync {
    async fn run(&self, request: WorkerRequest) -> Result<WorkerResponse, WorkerError>;
}

#[derive(Clone)]
pub struct RemoteWorkerClient {
    client: Client,
    endpoint: Url,
    bearer_token: Option<Arc<str>>,
    timeout: Duration,
}

impl RemoteWorkerClient {
    pub fn new(
        base_url: Url,
        bearer_token: Option<String>,
        timeout: Duration,
    ) -> Result<Self, WorkerError> {
        validate_worker_url(&base_url)?;
        let mut endpoint = base_url;
        endpoint.set_path("/v1/tasks");
        endpoint.set_query(None);
        endpoint.set_fragment(None);
        let client = Client::builder()
            .no_proxy()
            .redirect(Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .build()
            .map_err(WorkerError::BuildClient)?;
        Ok(Self {
            client,
            endpoint,
            bearer_token: bearer_token.map(Arc::from),
            timeout,
        })
    }

    async fn bounded_body(response: reqwest::Response) -> Result<Vec<u8>, WorkerError> {
        if response
            .content_length()
            .is_some_and(|length| length > MAX_WORKER_RESPONSE_BYTES as u64)
        {
            return Err(WorkerError::ResponseTooLarge);
        }

        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(WorkerError::Transport)?;
            if body.len().saturating_add(chunk.len()) > MAX_WORKER_RESPONSE_BYTES {
                return Err(WorkerError::ResponseTooLarge);
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body)
    }
}

#[async_trait]
impl WorkerBackend for RemoteWorkerClient {
    async fn run(&self, request: WorkerRequest) -> Result<WorkerResponse, WorkerError> {
        request.validate().map_err(WorkerError::InvalidRequest)?;
        let mut builder = self.client.post(self.endpoint.clone()).json(&request);
        if let Some(token) = &self.bearer_token {
            builder = builder.bearer_auth(token.as_ref());
        }

        let operation = async {
            let response = builder.send().await.map_err(WorkerError::Transport)?;
            let status = response.status();
            if status != StatusCode::OK {
                return Err(WorkerError::HttpStatus(status));
            }
            Self::bounded_body(response).await
        };
        let body = tokio::time::timeout(self.timeout, operation)
            .await
            .map_err(|_| WorkerError::Timeout)??;
        let response: WorkerResponse =
            serde_json::from_slice(&body).map_err(WorkerError::InvalidResponse)?;
        if response.request_id != request.request_id || response.run_id != request.run_id {
            return Err(WorkerError::MismatchedResponse);
        }
        response
            .result
            .validate()
            .map_err(WorkerError::InvalidRequest)?;
        Ok(response)
    }
}

pub(crate) fn validate_worker_url(url: &Url) -> Result<(), WorkerError> {
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(WorkerError::InvalidEndpointPolicy(
            "userinfo, query strings, and fragments are not allowed".into(),
        ));
    }
    let loopback = match url.host() {
        Some(Host::Ipv4(address)) => IpAddr::V4(address).is_loopback(),
        Some(Host::Ipv6(address)) => IpAddr::V6(address).is_loopback(),
        Some(Host::Domain(_)) | None => false,
    };
    if url.scheme() != "https" && !(url.scheme() == "http" && loopback) {
        return Err(WorkerError::InvalidEndpointPolicy(
            "remote workers require HTTPS; HTTP is allowed only for a literal loopback IP".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("invalid worker endpoint: {0}")]
    InvalidEndpoint(url::ParseError),
    #[error("invalid worker endpoint policy: {0}")]
    InvalidEndpointPolicy(String),
    #[error("could not build worker client: {0}")]
    BuildClient(reqwest::Error),
    #[error("worker request is invalid: {0}")]
    InvalidRequest(rlm_protocol::ValidationError),
    #[error("worker request timed out")]
    Timeout,
    #[error("worker transport failed: {0}")]
    Transport(reqwest::Error),
    #[error("worker returned HTTP {0}")]
    HttpStatus(StatusCode),
    #[error("worker response exceeded the 256 KiB protocol limit")]
    ResponseTooLarge,
    #[error("worker response was not valid protocol JSON: {0}")]
    InvalidResponse(serde_json::Error),
    #[error("worker response identifiers did not match the request")]
    MismatchedResponse,
}
