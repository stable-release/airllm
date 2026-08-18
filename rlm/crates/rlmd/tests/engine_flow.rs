use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use axum::{Json, Router, extract::State, routing::post};
use rlm_protocol::{
    ArtifactRef, ContextRef, RunCapabilities, RunEventKind, RunLimits, RunRequest, RunSnapshot,
    RunStatus, WorkerRequest, WorkerResponse,
};
use rlmd::{
    engine::{EngineError, RlmEngine},
    executor::{WorkerBackend, WorkerError},
    model::{ChatMessage, MlxModelClient, ModelBackend, ModelError, ModelReply, ModelUsage},
    store::RunStore,
};
use serde_json::json;
use tempfile::TempDir;
use tokio::{
    net::TcpListener,
    sync::Semaphore,
    task::JoinHandle,
    time::{sleep, timeout},
};
use url::Url;
use uuid::Uuid;

const TEST_TIMEOUT: Duration = Duration::from_secs(3);

struct ScriptedModel {
    replies: Mutex<VecDeque<ModelReply>>,
    calls: AtomicUsize,
    messages: Mutex<Vec<Vec<ChatMessage>>>,
    max_tokens: Mutex<Vec<u32>>,
}

impl ScriptedModel {
    fn new(replies: impl IntoIterator<Item = ModelReply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            calls: AtomicUsize::new(0),
            messages: Mutex::new(Vec::new()),
            max_tokens: Mutex::new(Vec::new()),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn captured_messages(&self) -> Vec<Vec<ChatMessage>> {
        self.messages.lock().unwrap().clone()
    }
}

#[async_trait]
impl ModelBackend for ScriptedModel {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: u32,
    ) -> Result<ModelReply, ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.messages.lock().unwrap().push(messages.to_vec());
        self.max_tokens.lock().unwrap().push(max_tokens);
        self.replies.lock().unwrap().pop_front().ok_or_else(|| {
            ModelError::MalformedResponse("scripted model ran out of replies".to_owned())
        })
    }

    async fn health(&self) -> Result<(), ModelError> {
        Ok(())
    }

    async fn readiness(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

#[derive(Default)]
struct RejectingWorker {
    calls: AtomicUsize,
}

#[async_trait]
impl WorkerBackend for RejectingWorker {
    async fn run(&self, _request: WorkerRequest) -> Result<WorkerResponse, WorkerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(WorkerError::Timeout)
    }
}

struct BlockingModel {
    entered: Semaphore,
    release: Semaphore,
    exited: Semaphore,
    calls: AtomicUsize,
}

impl BlockingModel {
    fn new() -> Self {
        Self {
            entered: Semaphore::new(0),
            release: Semaphore::new(0),
            exited: Semaphore::new(0),
            calls: AtomicUsize::new(0),
        }
    }

    async fn wait_until_entered(&self) {
        timeout(TEST_TIMEOUT, self.entered.acquire())
            .await
            .expect("model call did not start")
            .expect("entry semaphore was closed")
            .forget();
    }

    async fn wait_until_exited(&self) {
        timeout(TEST_TIMEOUT, self.exited.acquire())
            .await
            .expect("model call did not exit")
            .expect("exit semaphore was closed")
            .forget();
    }
}

#[async_trait]
impl ModelBackend for BlockingModel {
    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _max_tokens: u32,
    ) -> Result<ModelReply, ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.add_permits(1);
        self.release
            .acquire()
            .await
            .expect("release semaphore was closed")
            .forget();
        self.exited.add_permits(1);
        Ok(reply(r#"{"type":"final","answer":"released"}"#, 1, 1))
    }

    async fn health(&self) -> Result<(), ModelError> {
        Ok(())
    }

    async fn readiness(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

fn reply(content: &str, prompt_tokens: u64, completion_tokens: u64) -> ModelReply {
    ModelReply {
        content: content.to_owned(),
        usage: ModelUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        finish_reason: Some("stop".to_owned()),
    }
}

fn request(prompt: &str) -> RunRequest {
    RunRequest {
        prompt: prompt.to_owned(),
        contexts: Vec::new(),
        capabilities: RunCapabilities::default(),
        limits: RunLimits::default(),
    }
}

fn context() -> ContextRef {
    ContextRef {
        context_id: "known-context".to_owned(),
        artifact: ArtifactRef {
            artifact_id: "known-artifact".to_owned(),
            store: "remote".to_owned(),
            key: "documents/context.txt".to_owned(),
            media_type: Some("text/plain".to_owned()),
            sha256: None,
            size_bytes: Some(128),
        },
        description: Some("fixture context".to_owned()),
    }
}

fn test_engine(
    model: Arc<dyn ModelBackend>,
    worker: Option<Arc<dyn WorkerBackend>>,
    max_runs: usize,
) -> (TempDir, Arc<RlmEngine>) {
    let directory = tempfile::tempdir().unwrap();
    let store = RunStore::open(directory.path().join("state.sqlite3")).unwrap();
    let engine = Arc::new(RlmEngine::new(model, worker, store, max_runs).unwrap());
    (directory, engine)
}

async fn wait_for_status(
    engine: &RlmEngine,
    run_id: Uuid,
    predicate: impl Fn(RunStatus) -> bool,
) -> RunSnapshot {
    timeout(TEST_TIMEOUT, async {
        loop {
            let snapshot = engine
                .get_run(run_id)
                .expect("run lookup failed")
                .expect("run disappeared");
            if predicate(snapshot.status) {
                return snapshot;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("run did not reach the expected status")
}

async fn wait_for_terminal(engine: &RlmEngine, run_id: Uuid) -> RunSnapshot {
    wait_for_status(engine, run_id, RunStatus::is_terminal).await
}

#[tokio::test]
async fn final_only_run_succeeds_with_usage_and_contiguous_events() {
    let model = Arc::new(ScriptedModel::new([reply(
        r#"{"type":"final","answer":"offline answer"}"#,
        17,
        4,
    )]));
    let (_directory, engine) = test_engine(model.clone(), None, 1);

    let started = engine.start_run(request("answer locally")).await.unwrap();
    let finished = wait_for_terminal(&engine, started.run_id).await;

    assert_eq!(finished.status, RunStatus::Succeeded);
    assert_eq!(finished.final_answer.as_deref(), Some("offline answer"));
    assert_eq!(finished.usage.model_calls, 1);
    assert_eq!(finished.usage.input_tokens, 17);
    assert_eq!(finished.usage.output_tokens, 4);
    assert_eq!(finished.usage.worker_calls, 0);
    assert_eq!(model.call_count(), 1);

    let events = engine.list_events(started.run_id, None, 100).unwrap();
    assert_eq!(finished.next_sequence, events.len() as u64);
    for (expected, event) in events.iter().enumerate() {
        assert_eq!(event.sequence, expected as u64);
        event
            .validate_after(expected.checked_sub(1).map(|value| value as u64))
            .unwrap();
    }
    assert!(matches!(events[0].kind, RunEventKind::RunQueued));
    assert!(
        events
            .iter()
            .any(|event| matches!(event.kind, RunEventKind::RunStarted))
    );
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        RunEventKind::ActionRequested {
            action: rlm_protocol::ModelAction::Final { answer },
            ..
        } if answer == "offline answer"
    )));
    assert!(matches!(
        events.last().map(|event| &event.kind),
        Some(RunEventKind::RunCompleted { final_answer, usage })
            if final_answer == "offline answer" && usage.model_calls == 1
    ));
}

#[tokio::test]
async fn subcall_result_is_observed_before_the_root_model_finishes() {
    let model = Arc::new(ScriptedModel::new([
        reply(
            r#"{"type":"subcall","prompt":"compute the fact","context_ids":[],"max_output_tokens":64}"#,
            10,
            2,
        ),
        reply("the focused result is 42", 3, 5),
        reply(r#"{"type":"final","answer":"The result is 42."}"#, 12, 3),
    ]));
    let (_directory, engine) = test_engine(model.clone(), None, 1);

    let started = engine.start_run(request("use a subcall")).await.unwrap();
    let finished = wait_for_terminal(&engine, started.run_id).await;

    assert_eq!(finished.status, RunStatus::Succeeded);
    assert_eq!(finished.final_answer.as_deref(), Some("The result is 42."));
    assert_eq!(finished.usage.model_calls, 3);
    assert_eq!(finished.usage.input_tokens, 25);
    assert_eq!(finished.usage.output_tokens, 10);

    let calls = model.captured_messages();
    assert_eq!(calls.len(), 3);
    assert!(calls[1][0].content.contains("focused subquestion"));
    assert!(calls[1][1].content.contains("compute the fact"));
    assert!(
        calls[2]
            .last()
            .unwrap()
            .content
            .contains("the focused result is 42")
    );

    let events = engine.list_events(started.run_id, None, 100).unwrap();
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        RunEventKind::ActionCompleted { step: 0, usage, .. }
            if usage.model_calls == 1 && usage.output_tokens == 5
    )));
}

#[tokio::test]
async fn internet_search_is_rejected_before_any_model_call_or_run_creation() {
    let model = Arc::new(ScriptedModel::new([]));
    let (_directory, engine) = test_engine(model.clone(), None, 1);
    let mut run = request("search the internet");
    run.capabilities.internet_search = true;

    let error = engine.start_run(run).await.unwrap_err();

    assert!(matches!(error, EngineError::InternetUnavailable));
    assert_eq!(model.call_count(), 0);
}

#[tokio::test]
async fn remote_capabilities_are_rejected_when_no_worker_is_configured() {
    for capabilities in [
        RunCapabilities {
            remote_context: true,
            ..RunCapabilities::default()
        },
        RunCapabilities {
            remote_exec: true,
            ..RunCapabilities::default()
        },
    ] {
        let model = Arc::new(ScriptedModel::new([]));
        let (_directory, engine) = test_engine(model.clone(), None, 1);
        let mut run = request("use a remote capability");
        run.capabilities = capabilities;

        let error = engine.start_run(run).await.unwrap_err();

        assert!(matches!(error, EngineError::WorkerUnavailable));
        assert_eq!(model.call_count(), 0);
    }
}

#[tokio::test]
async fn fabricated_context_and_artifact_ids_are_rejected_without_reaching_the_worker() {
    let cases = [
        (
            r#"{"type":"read_context","context_id":"invented-context","offset":0,"length":64}"#,
            "unknown context `invented-context`",
        ),
        (
            r#"{"type":"execute_python","code":"print('x')","input_artifact_ids":["invented-artifact"]}"#,
            "unknown artifact `invented-artifact`",
        ),
    ];

    for (action, expected_error) in cases {
        let model = Arc::new(ScriptedModel::new([
            reply(action, 5, 2),
            reply(
                r#"{"type":"final","answer":"recovered without execution"}"#,
                6,
                2,
            ),
        ]));
        let worker = Arc::new(RejectingWorker::default());
        let (_directory, engine) = test_engine(model, Some(worker.clone()), 1);
        let mut run = request("try an invented identifier");
        run.contexts = vec![context()];
        run.capabilities = RunCapabilities {
            remote_context: true,
            remote_exec: true,
            internet_search: false,
        };

        let started = engine.start_run(run).await.unwrap();
        let finished = wait_for_terminal(&engine, started.run_id).await;

        assert_eq!(finished.status, RunStatus::Succeeded);
        assert_eq!(
            finished.final_answer.as_deref(),
            Some("recovered without execution")
        );
        assert_eq!(worker.calls.load(Ordering::SeqCst), 0);
        let events = engine.list_events(started.run_id, None, 100).unwrap();
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            RunEventKind::ActionRejected { step: 0, error }
                if error.contains(expected_error)
        )));
        assert!(!events.iter().any(|event| matches!(
            &event.kind,
            RunEventKind::ActionRequested {
                action: rlm_protocol::ModelAction::ReadContext { .. }
                    | rlm_protocol::ModelAction::ExecutePython { .. },
                ..
            }
        )));
    }
}

#[tokio::test]
async fn admission_is_bounded_and_running_cancellation_is_idempotent() {
    let model = Arc::new(BlockingModel::new());
    let (_directory, engine) = test_engine(model.clone(), None, 1);

    let running = engine
        .start_run(request("block in the model"))
        .await
        .unwrap();
    model.wait_until_entered().await;
    wait_for_status(&engine, running.run_id, |status| {
        status == RunStatus::Running
    })
    .await;

    let overload = engine
        .start_run(request("wait for the run slot"))
        .await
        .unwrap_err();
    assert!(matches!(overload, EngineError::Overloaded));

    let running_cancelled = engine.cancel_run(running.run_id).await.unwrap();
    assert_eq!(running_cancelled.status, RunStatus::Cancelled);
    let running_cancelled_again = engine.cancel_run(running.run_id).await.unwrap();
    assert_eq!(running_cancelled_again.status, RunStatus::Cancelled);

    assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    model.release.add_permits(1);
    model.wait_until_exited().await;
    sleep(Duration::from_millis(20)).await;

    assert_eq!(
        engine.get_run(running.run_id).unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(model.calls.load(Ordering::SeqCst), 1);

    let events = engine.list_events(running.run_id, None, 100).unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event.kind, RunEventKind::RunCancelled { .. }))
            .count(),
        1
    );
    assert!(matches!(
        events.last().map(|event| &event.kind),
        Some(RunEventKind::RunCancelled { .. })
    ));
}

#[derive(Clone)]
struct FakeMlxState {
    calls: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
    entered: Arc<Semaphore>,
    release_first: Arc<Semaphore>,
}

impl FakeMlxState {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
            entered: Arc::new(Semaphore::new(0)),
            release_first: Arc::new(Semaphore::new(0)),
        }
    }
}

async fn fake_mlx_completion(State(state): State<FakeMlxState>) -> Json<serde_json::Value> {
    let call_index = state.calls.fetch_add(1, Ordering::SeqCst);
    let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
    state.max_active.fetch_max(active, Ordering::SeqCst);
    state.entered.add_permits(1);
    if call_index == 0 {
        state
            .release_first
            .acquire()
            .await
            .expect("release semaphore was closed")
            .forget();
    }
    state.active.fetch_sub(1, Ordering::SeqCst);

    Json(json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "{\"type\":\"final\",\"answer\":\"served locally\"}"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5
        }
    }))
}

async fn spawn_fake_mlx(state: FakeMlxState) -> (Url, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = Router::new()
        .route("/v1/chat/completions", post(fake_mlx_completion))
        .with_state(state);
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (Url::parse(&format!("http://{address}")).unwrap(), server)
}

#[tokio::test]
async fn real_mlx_client_serializes_runs_and_does_not_release_on_engine_cancellation() {
    let state = FakeMlxState::new();
    let (base_url, server) = spawn_fake_mlx(state.clone()).await;
    let model = Arc::new(MlxModelClient::new(base_url, Duration::from_secs(2)).unwrap());
    let (_directory, engine) = test_engine(model, None, 2);

    let first = engine.start_run(request("first")).await.unwrap();
    timeout(TEST_TIMEOUT, state.entered.acquire())
        .await
        .expect("first HTTP model request did not start")
        .unwrap()
        .forget();
    assert_eq!(
        wait_for_status(&engine, first.run_id, |status| status == RunStatus::Running)
            .await
            .status,
        RunStatus::Running
    );

    assert_eq!(
        engine.cancel_run(first.run_id).await.unwrap().status,
        RunStatus::Cancelled
    );
    let second = engine.start_run(request("second")).await.unwrap();
    wait_for_status(&engine, second.run_id, |status| {
        status == RunStatus::Running
    })
    .await;

    assert!(
        timeout(Duration::from_millis(75), state.entered.acquire())
            .await
            .is_err(),
        "a second MLX request started before the cancelled request drained"
    );
    assert_eq!(state.calls.load(Ordering::SeqCst), 1);

    state.release_first.add_permits(1);
    timeout(TEST_TIMEOUT, state.entered.acquire())
        .await
        .expect("second HTTP model request did not start after the first drained")
        .unwrap()
        .forget();

    let first_finished = wait_for_terminal(&engine, first.run_id).await;
    let second_finished = wait_for_terminal(&engine, second.run_id).await;
    assert_eq!(first_finished.status, RunStatus::Cancelled);
    assert_eq!(second_finished.status, RunStatus::Succeeded);
    assert_eq!(
        second_finished.final_answer.as_deref(),
        Some("served locally")
    );
    assert_eq!(state.calls.load(Ordering::SeqCst), 2);
    assert_eq!(state.max_active.load(Ordering::SeqCst), 1);
    assert_eq!(state.active.load(Ordering::SeqCst), 0);

    server.abort();
}
