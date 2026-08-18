use std::{collections::VecDeque, convert::Infallible, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures_util::stream::{self, Stream};
use rlm_protocol::{RunEvent, RunRequest, RunSnapshot};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower_http::{
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::TraceLayer,
};
use tracing::{error, warn};
use uuid::Uuid;

use crate::engine::{EngineError, RlmEngine};

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<RlmEngine>,
    api_token: Option<Arc<str>>,
    event_stream_slots: Arc<Semaphore>,
}

impl AppState {
    pub fn new(engine: Arc<RlmEngine>, api_token: Option<String>) -> Self {
        Self {
            engine,
            api_token: api_token.map(Arc::from),
            event_stream_slots: Arc::new(Semaphore::new(32)),
        }
    }
}

pub fn router(state: AppState, request_bytes: usize) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/readyz", get(readiness))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/runs", post(create_run))
        .route("/v1/runs/{run_id}", get(get_run))
        .route("/v1/runs/{run_id}/cancel", post(cancel_run))
        .route("/v1/runs/{run_id}/events", get(run_events))
        .layer(DefaultBodyLimit::max(request_bytes))
        .layer(TraceLayer::new_for_http())
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        inference_mode: "offline",
    })
}

async fn readiness(State(state): State<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    if let Err(error) = state.engine.model_readiness().await {
        warn!(%error, "model readiness check failed");
        return Err(ApiError::unavailable(
            "model_not_ready",
            "local model is not ready",
        ));
    }
    Ok(Json(HealthResponse {
        status: "ready",
        inference_mode: "offline",
    }))
}

async fn capabilities(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<CapabilitiesResponse>, ApiError> {
    authorize(&state, &headers)?;
    Ok(Json(CapabilitiesResponse {
        local_model: true,
        remote_context: state.engine.worker_available(),
        remote_exec: state.engine.worker_available(),
        internet_search: false,
    }))
}

async fn create_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RunRequest>,
) -> Result<(StatusCode, Json<RunSnapshot>), ApiError> {
    authorize(&state, &headers)?;
    let snapshot = state
        .engine
        .start_run(request)
        .await
        .map_err(ApiError::from)?;
    Ok((StatusCode::ACCEPTED, Json(snapshot)))
}

async fn get_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(run_id): Path<Uuid>,
) -> Result<Json<RunSnapshot>, ApiError> {
    authorize(&state, &headers)?;
    state
        .engine
        .get_run(run_id)
        .map_err(ApiError::from)?
        .map(Json)
        .ok_or_else(|| ApiError::not_found("run_not_found", "run was not found"))
}

async fn cancel_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(run_id): Path<Uuid>,
) -> Result<Json<RunSnapshot>, ApiError> {
    authorize(&state, &headers)?;
    state
        .engine
        .cancel_run(run_id)
        .await
        .map(Json)
        .map_err(ApiError::from)
}

async fn run_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(run_id): Path<Uuid>,
    Query(query): Query<EventQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    authorize(&state, &headers)?;
    if state
        .engine
        .get_run(run_id)
        .map_err(ApiError::from)?
        .is_none()
    {
        return Err(ApiError::not_found("run_not_found", "run was not found"));
    }
    let permit = state
        .event_stream_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::too_many("event_stream_limit", "too many event streams"))?;
    let cursor = EventCursor {
        engine: state.engine,
        run_id,
        after: query.after,
        pending: VecDeque::new(),
        terminal: false,
        _permit: permit,
    };
    let stream = stream::unfold(cursor, next_sse_event);
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    ))
}

async fn next_sse_event(
    mut cursor: EventCursor,
) -> Option<(Result<Event, Infallible>, EventCursor)> {
    loop {
        if let Some(event) = cursor.pending.pop_front() {
            cursor.after = Some(event.sequence);
            let payload = serde_json::to_string(&event)
                .unwrap_or_else(|_| "{\"type\":\"serialization_error\"}".into());
            let sse = Event::default()
                .event("run_event")
                .id(event.sequence.to_string())
                .data(payload);
            return Some((Ok(sse), cursor));
        }
        if cursor.terminal {
            return None;
        }

        match cursor.engine.list_events(cursor.run_id, cursor.after, 16) {
            Ok(events) if !events.is_empty() => {
                cursor.pending.extend(events);
                continue;
            }
            Ok(_) => match cursor.engine.get_run(cursor.run_id) {
                Ok(Some(snapshot)) if snapshot.status.is_terminal() => {
                    if !terminal_events_consumed(cursor.after, snapshot.next_sequence) {
                        continue;
                    }
                    cursor.terminal = true;
                    continue;
                }
                Ok(Some(_)) => {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    continue;
                }
                Ok(None) => return None,
                Err(error) => {
                    error!(%error, run_id = %cursor.run_id, "event stream failed");
                    return None;
                }
            },
            Err(error) => {
                error!(%error, run_id = %cursor.run_id, "event stream failed");
                return None;
            }
        }
    }
}

fn terminal_events_consumed(after: Option<u64>, next_sequence: u64) -> bool {
    let consumed = after.map_or(0, |sequence| sequence.saturating_add(1));
    consumed >= next_sequence
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    authorize_token(state.api_token.as_deref(), headers)
}

fn authorize_token(expected: Option<&str>, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return Err(ApiError::unauthorized());
    };
    let Ok(value) = value.to_str() else {
        return Err(ApiError::unauthorized());
    };
    let Some(provided) = value.strip_prefix("Bearer ") else {
        return Err(ApiError::unauthorized());
    };
    let equal: bool =
        provided.len() == expected.len() && provided.as_bytes().ct_eq(expected.as_bytes()).into();
    if equal {
        Ok(())
    } else {
        Err(ApiError::unauthorized())
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    inference_mode: &'static str,
}

#[derive(Serialize)]
struct CapabilitiesResponse {
    local_model: bool,
    remote_context: bool,
    remote_exec: bool,
    internet_search: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EventQuery {
    after: Option<u64>,
}

struct EventCursor {
    engine: Arc<RlmEngine>,
    run_id: Uuid,
    after: Option<u64>,
    pending: VecDeque<RunEvent>,
    terminal: bool,
    _permit: OwnedSemaphorePermit,
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
            message: "a valid bearer token is required".into(),
        }
    }

    fn not_found(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code,
            message: message.into(),
        }
    }

    fn unavailable(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code,
            message: message.into(),
        }
    }

    fn too_many(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            code,
            message: message.into(),
        }
    }
}

impl From<EngineError> for ApiError {
    fn from(error: EngineError) -> Self {
        let (status, code, expose) = match &error {
            EngineError::InvalidRequest(_)
            | EngineError::Validation(_)
            | EngineError::InternetUnavailable
            | EngineError::PayloadTooLarge(_, _)
            | EngineError::TooManyContexts(_) => (StatusCode::BAD_REQUEST, "invalid_request", true),
            EngineError::RunNotFound(_) => (StatusCode::NOT_FOUND, "run_not_found", true),
            EngineError::WorkerUnavailable => {
                (StatusCode::SERVICE_UNAVAILABLE, "worker_unavailable", true)
            }
            EngineError::Overloaded => (StatusCode::TOO_MANY_REQUESTS, "overloaded", true),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", false),
        };
        if !expose {
            error!(%error, "API request failed");
        }
        Self {
            status,
            code,
            message: if expose {
                error.to_string()
            } else {
                "the request could not be completed".into()
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response<Body> {
        let mut response = (
            self.status,
            Json(ErrorBody {
                error: ErrorDetail {
                    code: self.code,
                    message: self.message,
                },
            }),
        )
            .into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                header::HeaderValue::from_static("Bearer"),
            );
        }
        response
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    code: &'static str,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_auth_is_fail_closed_and_constant_length_aware() {
        let mut headers = HeaderMap::new();
        assert!(authorize_token(Some("secret"), &headers).is_err());
        headers.insert(header::AUTHORIZATION, "Bearer wrong".parse().unwrap());
        assert!(authorize_token(Some("secret"), &headers).is_err());
        headers.insert(header::AUTHORIZATION, "Bearer secret".parse().unwrap());
        assert!(authorize_token(Some("secret"), &headers).is_ok());
        assert!(authorize_token(None, &HeaderMap::new()).is_ok());
    }

    #[test]
    fn terminal_stream_waits_for_every_committed_event() {
        assert!(!terminal_events_consumed(None, 2));
        assert!(!terminal_events_consumed(Some(0), 2));
        assert!(terminal_events_consumed(Some(1), 2));
        assert!(terminal_events_consumed(None, 0));
    }
}
