use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::State,
    http::{Request, StatusCode, header},
    response::Redirect,
    routing::{any, post},
};
use chrono::{TimeDelta, Utc};
use rlm_protocol::{WorkerOperation, WorkerRequest, WorkerResponse, WorkerResult, WorkerUsage};
use rlmd::{
    engine::RlmEngine,
    executor::{RemoteWorkerClient, WorkerBackend, WorkerError},
    model::{ChatMessage, ModelBackend, ModelError, ModelReply, ModelUsage},
    server::{AppState, router},
    store::RunStore,
};
use serde_json::{Value, json};
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;
use url::Url;
use uuid::Uuid;

#[derive(Debug)]
struct ImmediateModel;

#[async_trait]
impl ModelBackend for ImmediateModel {
    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _max_tokens: u32,
    ) -> Result<ModelReply, ModelError> {
        Ok(ModelReply {
            content: r#"{"type":"final","answer":"offline"}"#.into(),
            usage: ModelUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            finish_reason: Some("stop".into()),
        })
    }

    async fn health(&self) -> Result<(), ModelError> {
        Ok(())
    }

    async fn readiness(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

fn api(api_token: Option<&str>) -> Router {
    let model: Arc<dyn ModelBackend> = Arc::new(ImmediateModel);
    let store = RunStore::open(":memory:").expect("open in-memory run store");
    let engine =
        Arc::new(RlmEngine::new(model, None, store, 1).expect("construct test RLM engine"));
    router(
        AppState::new(engine, api_token.map(str::to_owned)),
        64 * 1024,
    )
}

async fn json_body(response: axum::response::Response) -> Value {
    let body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("read response body");
    serde_json::from_slice(&body).expect("response contains JSON")
}

#[tokio::test]
async fn health_is_public_but_v1_is_authenticated_and_offline_by_default() {
    let app = api(Some("correct-horse"));

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(
        json_body(health).await,
        json!({"status": "ok", "inference_mode": "offline"})
    );

    let unauthorized = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        unauthorized.headers().get(header::WWW_AUTHENTICATE),
        Some(&header::HeaderValue::from_static("Bearer"))
    );

    let capabilities = app
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .header(header::AUTHORIZATION, "Bearer correct-horse")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(capabilities.status(), StatusCode::OK);
    assert_eq!(
        json_body(capabilities).await,
        json!({
            "local_model": true,
            "remote_context": false,
            "remote_exec": false,
            "internet_search": false
        })
    );
}

#[tokio::test]
async fn run_api_rejects_unknown_fields_and_internet_search() {
    let app = api(Some("secret"));

    let unknown_field = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/runs")
                .header(header::AUTHORIZATION, "Bearer secret")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"prompt":"stay offline","model_url":"https://example.com"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(unknown_field.status().is_client_error());

    let internet = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/runs")
                .header(header::AUTHORIZATION, "Bearer secret")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"prompt":"search for this","capabilities":{"internet_search":true}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(internet.status(), StatusCode::BAD_REQUEST);
    let body = json_body(internet).await;
    assert_eq!(body["error"]["code"], "invalid_request");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("internet search"))
    );
}

#[derive(Clone, Default)]
struct WorkerRecorder {
    task_hits: Arc<AtomicUsize>,
    other_hits: Arc<AtomicUsize>,
    redirect_target_hits: Arc<AtomicUsize>,
}

async fn spawn_mock_worker(app: Router) -> (Url, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock worker");
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve mock worker");
    });
    (Url::parse(&format!("http://{address}/")).unwrap(), task)
}

fn worker_request() -> WorkerRequest {
    WorkerRequest {
        request_id: Uuid::new_v4(),
        run_id: Uuid::new_v4(),
        deadline: Utc::now() + TimeDelta::seconds(10),
        operation: WorkerOperation::ExecutePython {
            code: "print(2 + 2)".into(),
            input_artifacts: Vec::new(),
            timeout_ms: 1_000,
        },
    }
}

async fn successful_task(
    State(recorder): State<WorkerRecorder>,
    Json(request): Json<WorkerRequest>,
) -> Json<WorkerResponse> {
    recorder.task_hits.fetch_add(1, Ordering::SeqCst);
    Json(WorkerResponse {
        request_id: request.request_id,
        run_id: request.run_id,
        result: WorkerResult::Succeeded {
            output: Some("4".into()),
            artifacts: Vec::new(),
            usage: WorkerUsage::default(),
        },
    })
}

async fn other_endpoint(State(recorder): State<WorkerRecorder>) -> StatusCode {
    recorder.other_hits.fetch_add(1, Ordering::SeqCst);
    StatusCode::NOT_FOUND
}

#[tokio::test]
async fn remote_worker_uses_only_the_configured_task_endpoint() {
    let recorder = WorkerRecorder::default();
    let app = Router::new()
        .route("/v1/tasks", post(successful_task))
        .fallback(any(other_endpoint))
        .with_state(recorder.clone());
    let (base_url, server) = spawn_mock_worker(app).await;
    let client = RemoteWorkerClient::new(base_url, None, Duration::from_secs(2)).unwrap();

    let response = client.run(worker_request()).await.unwrap();
    assert!(matches!(response.result, WorkerResult::Succeeded { .. }));
    assert_eq!(recorder.task_hits.load(Ordering::SeqCst), 1);
    assert_eq!(recorder.other_hits.load(Ordering::SeqCst), 0);

    server.abort();
}

async fn mismatched_task(Json(request): Json<WorkerRequest>) -> Json<WorkerResponse> {
    Json(WorkerResponse {
        request_id: Uuid::new_v4(),
        run_id: request.run_id,
        result: WorkerResult::Succeeded {
            output: None,
            artifacts: Vec::new(),
            usage: WorkerUsage::default(),
        },
    })
}

#[tokio::test]
async fn remote_worker_rejects_mismatched_response_identifiers() {
    let app = Router::new().route("/v1/tasks", post(mismatched_task));
    let (base_url, server) = spawn_mock_worker(app).await;
    let client = RemoteWorkerClient::new(base_url, None, Duration::from_secs(2)).unwrap();

    let error = client.run(worker_request()).await.unwrap_err();
    assert!(matches!(error, WorkerError::MismatchedResponse));

    server.abort();
}

async fn redirect_task(State(recorder): State<WorkerRecorder>) -> Redirect {
    recorder.task_hits.fetch_add(1, Ordering::SeqCst);
    Redirect::temporary("/redirect-target")
}

async fn redirect_target(State(recorder): State<WorkerRecorder>) -> StatusCode {
    recorder.redirect_target_hits.fetch_add(1, Ordering::SeqCst);
    StatusCode::OK
}

#[tokio::test]
async fn remote_worker_does_not_follow_redirects() {
    let recorder = WorkerRecorder::default();
    let app = Router::new()
        .route("/v1/tasks", post(redirect_task))
        .route("/redirect-target", post(redirect_target))
        .with_state(recorder.clone());
    let (base_url, server) = spawn_mock_worker(app).await;
    let client = RemoteWorkerClient::new(base_url, None, Duration::from_secs(2)).unwrap();

    let error = client.run(worker_request()).await.unwrap_err();
    assert!(matches!(error, WorkerError::HttpStatus(status) if status.is_redirection()));
    assert_eq!(recorder.task_hits.load(Ordering::SeqCst), 1);
    assert_eq!(recorder.redirect_target_hits.load(Ordering::SeqCst), 0);

    server.abort();
}
