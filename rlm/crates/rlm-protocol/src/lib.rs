//! Shared wire types for the RLM coordinator, model adapter, and remote workers.
//!
//! The protocol is offline-first. Capabilities that can cross a trust boundary are
//! explicit and default to denied. References contain opaque storage coordinates,
//! never inline file or media payloads.

use std::{collections::HashSet, fmt};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub const DEFAULT_MAX_STEPS: u32 = 8;
pub const DEFAULT_MAX_SUBCALLS: u32 = 6;
pub const DEFAULT_MAX_DEPTH: u16 = 1;
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 6_000;
pub const DEFAULT_MAX_WALL_TIME_MS: u64 = 300_000;
pub const DEFAULT_MAX_ACTION_TIME_MS: u64 = 60_000;

pub const MAX_STEPS: u32 = 128;
pub const MAX_SUBCALLS: u32 = 64;
pub const MAX_DEPTH: u16 = 8;
pub const MAX_OUTPUT_TOKENS: u32 = 131_072;
pub const MAX_WALL_TIME_MS: u64 = 3_600_000;
pub const MAX_ACTION_TIME_MS: u64 = 600_000;
pub const MAX_ACTION_IDS: usize = 64;

/// Capabilities granted to one run.
///
/// `internet_search` reserves the wire-level permission for a future search MCP.
/// This protocol does not define an internet action, and v0 servers are expected
/// to reject the capability even when a client requests it.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunCapabilities {
    pub remote_context: bool,
    pub remote_exec: bool,
    pub internet_search: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunLimits {
    pub max_steps: u32,
    pub max_subcalls: u32,
    pub max_depth: u16,
    pub max_output_tokens: u32,
    pub max_wall_time_ms: u64,
    pub max_action_time_ms: u64,
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            max_subcalls: DEFAULT_MAX_SUBCALLS,
            max_depth: DEFAULT_MAX_DEPTH,
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            max_wall_time_ms: DEFAULT_MAX_WALL_TIME_MS,
            max_action_time_ms: DEFAULT_MAX_ACTION_TIME_MS,
        }
    }
}

impl RunLimits {
    /// Validates only scalar policy limits; transport payload limits are enforced
    /// separately so this method never serializes or copies request data.
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_range("max_steps", self.max_steps as u64, 1, MAX_STEPS as u64)?;
        validate_range(
            "max_subcalls",
            self.max_subcalls as u64,
            0,
            MAX_SUBCALLS as u64,
        )?;
        validate_range("max_depth", self.max_depth as u64, 0, MAX_DEPTH as u64)?;
        validate_range(
            "max_output_tokens",
            self.max_output_tokens as u64,
            1,
            MAX_OUTPUT_TOKENS as u64,
        )?;
        validate_range(
            "max_wall_time_ms",
            self.max_wall_time_ms,
            1,
            MAX_WALL_TIME_MS,
        )?;
        validate_range(
            "max_action_time_ms",
            self.max_action_time_ms,
            1,
            MAX_ACTION_TIME_MS,
        )?;

        if self.max_action_time_ms > self.max_wall_time_ms {
            return Err(ValidationError::InconsistentLimits {
                message: "max_action_time_ms must not exceed max_wall_time_ms",
            });
        }

        if self.max_depth == 0 && self.max_subcalls != 0 {
            return Err(ValidationError::InconsistentLimits {
                message: "max_subcalls must be zero when max_depth is zero",
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest {
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contexts: Vec<ContextRef>,
    #[serde(default)]
    pub capabilities: RunCapabilities,
    #[serde(default)]
    pub limits: RunLimits,
}

impl RunRequest {
    /// Performs semantic validation without measuring or copying payloads.
    pub fn validate(&self) -> Result<(), ValidationError> {
        require_non_blank("prompt", &self.prompt)?;
        self.limits.validate()?;

        if !self.contexts.is_empty() && !self.capabilities.remote_context {
            return Err(ValidationError::CapabilityRequired("remote_context"));
        }

        for (index, context) in self.contexts.iter().enumerate() {
            context
                .validate()
                .map_err(|source| ValidationError::AtIndex {
                    field: "contexts",
                    index,
                    source: Box::new(source),
                })?;
        }

        Ok(())
    }
}

/// Opaque reference to a context object held outside the model host.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextRef {
    pub context_id: String,
    pub artifact: ArtifactRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ContextRef {
    pub fn validate(&self) -> Result<(), ValidationError> {
        require_non_blank("context_id", &self.context_id)?;
        self.artifact.validate()?;
        if let Some(description) = &self.description {
            require_non_blank("description", description)?;
        }
        Ok(())
    }
}

/// Opaque coordinate for an artifact. `store` names a preconfigured backend and
/// `key` is interpreted only by that backend; it is deliberately not a URL.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub artifact_id: String,
    pub store: String,
    pub key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

impl ArtifactRef {
    pub fn validate(&self) -> Result<(), ValidationError> {
        require_non_blank("artifact_id", &self.artifact_id)?;
        require_non_blank("store", &self.store)?;
        require_non_blank("key", &self.key)?;
        if let Some(media_type) = &self.media_type {
            require_non_blank("media_type", media_type)?;
        }
        if let Some(sha256) = &self.sha256 {
            if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(ValidationError::InvalidSha256);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl RunStatus {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Queued, Self::Running | Self::Cancelled | Self::Failed)
                | (
                    Self::Running,
                    Self::Succeeded | Self::Failed | Self::Cancelled
                )
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Usage {
    pub model_calls: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub worker_calls: u32,
    pub wall_time_ms: u64,
}

impl Usage {
    pub fn saturating_add_assign(&mut self, other: Self) {
        self.model_calls = self.model_calls.saturating_add(other.model_calls);
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.worker_calls = self.worker_calls.saturating_add(other.worker_calls);
        self.wall_time_ms = self.wall_time_ms.saturating_add(other.wall_time_ms);
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunSnapshot {
    pub run_id: Uuid,
    pub status: RunStatus,
    /// The sequence number that will be assigned to the next event.
    pub next_sequence: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_answer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ArtifactRef>,
}

impl RunSnapshot {
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self.status {
            RunStatus::Succeeded if self.final_answer.is_none() => {
                return Err(ValidationError::InvalidSnapshot(
                    "a succeeded run requires final_answer",
                ));
            }
            RunStatus::Failed if self.error.is_none() => {
                return Err(ValidationError::InvalidSnapshot(
                    "a failed run requires error",
                ));
            }
            RunStatus::Queued | RunStatus::Running
                if self.final_answer.is_some() || self.error.is_some() =>
            {
                return Err(ValidationError::InvalidSnapshot(
                    "a non-terminal run cannot contain final_answer or error",
                ));
            }
            _ => {}
        }

        for (index, artifact) in self.artifacts.iter().enumerate() {
            artifact
                .validate()
                .map_err(|source| ValidationError::AtIndex {
                    field: "artifacts",
                    index,
                    source: Box::new(source),
                })?;
        }
        Ok(())
    }
}

/// One append-only run event. Events begin at sequence zero and are contiguous.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunEvent {
    pub run_id: Uuid,
    pub sequence: u64,
    pub occurred_at: DateTime<Utc>,
    pub kind: RunEventKind,
}

impl RunEvent {
    /// Verifies this event is the next contiguous event. Pass `None` for the first
    /// event in a run, which must use sequence zero.
    pub fn validate_after(&self, previous: Option<u64>) -> Result<(), ValidationError> {
        let expected = match previous {
            Some(previous) => previous
                .checked_add(1)
                .ok_or(ValidationError::SequenceExhausted)?,
            None => 0,
        };
        if self.sequence != expected {
            return Err(ValidationError::UnexpectedSequence {
                expected,
                actual: self.sequence,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum RunEventKind {
    RunQueued,
    RunStarted,
    ActionRequested {
        step: u32,
        action: ModelAction,
    },
    ActionRejected {
        step: u32,
        error: String,
    },
    ActionCompleted {
        step: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        artifacts: Vec<ArtifactRef>,
        #[serde(default)]
        usage: Usage,
    },
    StatusChanged {
        from: RunStatus,
        to: RunStatus,
    },
    RunCompleted {
        final_answer: String,
        #[serde(default)]
        usage: Usage,
    },
    RunFailed {
        error: String,
    },
    RunCancelled {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

/// The complete set of actions the model can request in protocol v0.
/// There is intentionally no generic HTTP or internet action.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelAction {
    Final {
        answer: String,
    },
    Subcall {
        prompt: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        context_ids: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_output_tokens: Option<u32>,
    },
    ReadContext {
        context_id: String,
        offset: u64,
        length: u32,
    },
    ExecutePython {
        code: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        input_artifact_ids: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
    },
    SaveNote {
        note: String,
    },
}

impl ModelAction {
    /// Validates semantic policy without measuring or copying action payloads.
    pub fn validate_for(
        &self,
        capabilities: RunCapabilities,
        limits: RunLimits,
        current_depth: u16,
    ) -> Result<(), ValidationError> {
        limits.validate()?;
        match self {
            Self::Final { answer } => require_non_blank("answer", answer),
            Self::Subcall {
                prompt,
                context_ids,
                max_output_tokens,
            } => {
                require_non_blank("prompt", prompt)?;
                if current_depth >= limits.max_depth {
                    return Err(ValidationError::DepthExceeded {
                        current: current_depth,
                        maximum: limits.max_depth,
                    });
                }
                if !context_ids.is_empty() && !capabilities.remote_context {
                    return Err(ValidationError::CapabilityRequired("remote_context"));
                }
                if let Some(tokens) = max_output_tokens {
                    validate_range(
                        "max_output_tokens",
                        *tokens as u64,
                        1,
                        limits.max_output_tokens as u64,
                    )?;
                }
                for context_id in context_ids {
                    require_non_blank("context_ids", context_id)?;
                }
                validate_unique_ids("context_ids", context_ids)?;
                Ok(())
            }
            Self::ReadContext {
                context_id, length, ..
            } => {
                if !capabilities.remote_context {
                    return Err(ValidationError::CapabilityRequired("remote_context"));
                }
                require_non_blank("context_id", context_id)?;
                validate_range("length", *length as u64, 1, u32::MAX as u64)
            }
            Self::ExecutePython {
                code,
                input_artifact_ids,
                timeout_ms,
            } => {
                if !capabilities.remote_exec {
                    return Err(ValidationError::CapabilityRequired("remote_exec"));
                }
                require_non_blank("code", code)?;
                if !input_artifact_ids.is_empty() && !capabilities.remote_context {
                    return Err(ValidationError::CapabilityRequired("remote_context"));
                }
                if let Some(timeout_ms) = timeout_ms {
                    validate_range("timeout_ms", *timeout_ms, 1, limits.max_action_time_ms)?;
                }
                for artifact_id in input_artifact_ids {
                    require_non_blank("input_artifact_ids", artifact_id)?;
                }
                validate_unique_ids("input_artifact_ids", input_artifact_ids)?;
                Ok(())
            }
            Self::SaveNote { note } => require_non_blank("note", note),
        }
    }
}

/// A coordinator request to a remote worker. The operation enum is closed so a
/// worker can never be tricked into making an arbitrary network request.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerRequest {
    pub request_id: Uuid,
    pub run_id: Uuid,
    pub deadline: DateTime<Utc>,
    pub operation: WorkerOperation,
}

impl WorkerRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.operation.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerOperation {
    ReadContext {
        context: ContextRef,
        offset: u64,
        length: u32,
    },
    ExecutePython {
        code: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        input_artifacts: Vec<ArtifactRef>,
        timeout_ms: u64,
    },
}

impl WorkerOperation {
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::ReadContext {
                context, length, ..
            } => {
                context.validate()?;
                validate_range("length", *length as u64, 1, u32::MAX as u64)
            }
            Self::ExecutePython {
                code,
                input_artifacts,
                timeout_ms,
            } => {
                require_non_blank("code", code)?;
                validate_range("timeout_ms", *timeout_ms, 1, MAX_ACTION_TIME_MS)?;
                for (index, artifact) in input_artifacts.iter().enumerate() {
                    artifact
                        .validate()
                        .map_err(|source| ValidationError::AtIndex {
                            field: "input_artifacts",
                            index,
                            source: Box::new(source),
                        })?;
                }
                validate_unique_artifacts("input_artifacts", input_artifacts)?;
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerResponse {
    pub request_id: Uuid,
    pub run_id: Uuid,
    pub result: WorkerResult,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerResult {
    Succeeded {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        artifacts: Vec<ArtifactRef>,
        #[serde(default)]
        usage: WorkerUsage,
    },
    Failed {
        code: WorkerErrorCode,
        message: String,
        #[serde(default)]
        retryable: bool,
    },
}

impl WorkerResult {
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Succeeded { artifacts, .. } => {
                for (index, artifact) in artifacts.iter().enumerate() {
                    artifact
                        .validate()
                        .map_err(|source| ValidationError::AtIndex {
                            field: "artifacts",
                            index,
                            source: Box::new(source),
                        })?;
                }
                validate_unique_artifacts("artifacts", artifacts)?;
                Ok(())
            }
            Self::Failed { message, .. } => require_non_blank("message", message),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerErrorCode {
    InvalidRequest,
    DeadlineExceeded,
    ExecutionFailed,
    ContextUnavailable,
    ResourceLimit,
    Cancelled,
    Internal,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerUsage {
    pub wall_time_ms: u64,
    pub cpu_time_ms: u64,
    pub peak_memory_bytes: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ValidationError {
    #[error("{field} must not be blank")]
    Blank { field: &'static str },
    #[error("{field} must be between {minimum} and {maximum}, got {actual}")]
    OutOfRange {
        field: &'static str,
        minimum: u64,
        maximum: u64,
        actual: u64,
    },
    #[error("inconsistent limits: {message}")]
    InconsistentLimits { message: &'static str },
    #[error("capability `{0}` is required")]
    CapabilityRequired(&'static str),
    #[error("maximum recursion depth {maximum} reached at depth {current}")]
    DepthExceeded { current: u16, maximum: u16 },
    #[error("invalid sha256 digest")]
    InvalidSha256,
    #[error("{field} contains {actual} entries; maximum is {maximum}")]
    TooManyItems {
        field: &'static str,
        maximum: usize,
        actual: usize,
    },
    #[error("{field} contains a duplicate identifier")]
    DuplicateIdentifier { field: &'static str },
    #[error("invalid {field}[{index}]: {source}")]
    AtIndex {
        field: &'static str,
        index: usize,
        source: Box<ValidationError>,
    },
    #[error("invalid snapshot: {0}")]
    InvalidSnapshot(&'static str),
    #[error("event sequence exhausted")]
    SequenceExhausted,
    #[error("expected event sequence {expected}, got {actual}")]
    UnexpectedSequence { expected: u64, actual: u64 },
}

fn require_non_blank(field: &'static str, value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError::Blank { field })
    } else {
        Ok(())
    }
}

fn validate_range(
    field: &'static str,
    actual: u64,
    minimum: u64,
    maximum: u64,
) -> Result<(), ValidationError> {
    if !(minimum..=maximum).contains(&actual) {
        Err(ValidationError::OutOfRange {
            field,
            minimum,
            maximum,
            actual,
        })
    } else {
        Ok(())
    }
}

fn validate_unique_ids(field: &'static str, ids: &[String]) -> Result<(), ValidationError> {
    if ids.len() > MAX_ACTION_IDS {
        return Err(ValidationError::TooManyItems {
            field,
            maximum: MAX_ACTION_IDS,
            actual: ids.len(),
        });
    }
    let mut unique = HashSet::with_capacity(ids.len());
    if ids.iter().any(|id| !unique.insert(id.as_str())) {
        return Err(ValidationError::DuplicateIdentifier { field });
    }
    Ok(())
}

fn validate_unique_artifacts(
    field: &'static str,
    artifacts: &[ArtifactRef],
) -> Result<(), ValidationError> {
    if artifacts.len() > MAX_ACTION_IDS {
        return Err(ValidationError::TooManyItems {
            field,
            maximum: MAX_ACTION_IDS,
            actual: artifacts.len(),
        });
    }
    let mut unique = HashSet::with_capacity(artifacts.len());
    if artifacts
        .iter()
        .any(|artifact| !unique.insert(artifact.artifact_id.as_str()))
    {
        return Err(ValidationError::DuplicateIdentifier { field });
    }
    Ok(())
}

impl fmt::Display for RunStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> ArtifactRef {
        ArtifactRef {
            artifact_id: "artifact-1".into(),
            store: "primary".into(),
            key: "runs/abc/context.txt".into(),
            media_type: Some("text/plain".into()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(42),
        }
    }

    fn context() -> ContextRef {
        ContextRef {
            context_id: "context-1".into(),
            artifact: artifact(),
            description: Some("test context".into()),
        }
    }

    #[test]
    fn omitted_capabilities_are_secure_and_offline() {
        let request: RunRequest = serde_json::from_str(r#"{"prompt":"hello"}"#).unwrap();

        assert_eq!(request.capabilities, RunCapabilities::default());
        assert!(!request.capabilities.remote_context);
        assert!(!request.capabilities.remote_exec);
        assert!(!request.capabilities.internet_search);
        assert_eq!(request.limits, RunLimits::default());
        request.validate().unwrap();
    }

    #[test]
    fn unknown_request_fields_are_rejected() {
        let error =
            serde_json::from_str::<RunRequest>(r#"{"prompt":"hello","allow_network":true}"#)
                .unwrap_err();

        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn contexts_require_an_explicit_capability() {
        let request = RunRequest {
            prompt: "inspect this".into(),
            contexts: vec![context()],
            capabilities: RunCapabilities::default(),
            limits: RunLimits::default(),
        };

        assert_eq!(
            request.validate(),
            Err(ValidationError::CapabilityRequired("remote_context"))
        );
    }

    #[test]
    fn model_actions_round_trip_as_tagged_json() {
        let action = ModelAction::ExecutePython {
            code: "print(6 * 7)".into(),
            input_artifact_ids: vec!["artifact-1".into()],
            timeout_ms: Some(5_000),
        };

        let json = serde_json::to_string(&action).unwrap();
        assert!(json.contains(r#""type":"execute_python""#));
        assert_eq!(serde_json::from_str::<ModelAction>(&json).unwrap(), action);
    }

    #[test]
    fn worker_messages_round_trip() {
        let request = WorkerRequest {
            request_id: Uuid::new_v4(),
            run_id: Uuid::new_v4(),
            deadline: Utc::now(),
            operation: WorkerOperation::ReadContext {
                context: context(),
                offset: 12,
                length: 100,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        let decoded: WorkerRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, request);
        decoded.validate().unwrap();
    }

    #[test]
    fn run_events_require_contiguous_monotonic_sequences() {
        let run_id = Uuid::new_v4();
        let first = RunEvent {
            run_id,
            sequence: 0,
            occurred_at: Utc::now(),
            kind: RunEventKind::RunQueued,
        };
        first.validate_after(None).unwrap();

        let next = RunEvent {
            run_id,
            sequence: 1,
            occurred_at: Utc::now(),
            kind: RunEventKind::RunStarted,
        };
        next.validate_after(Some(first.sequence)).unwrap();

        assert_eq!(
            next.validate_after(None),
            Err(ValidationError::UnexpectedSequence {
                expected: 0,
                actual: 1,
            })
        );
    }

    #[test]
    fn invalid_limits_are_rejected() {
        let too_many_steps = RunLimits {
            max_steps: MAX_STEPS + 1,
            ..RunLimits::default()
        };
        assert!(matches!(
            too_many_steps.validate(),
            Err(ValidationError::OutOfRange {
                field: "max_steps",
                ..
            })
        ));

        let action_exceeds_run = RunLimits {
            max_wall_time_ms: 10,
            max_action_time_ms: 11,
            ..RunLimits::default()
        };
        assert!(matches!(
            action_exceeds_run.validate(),
            Err(ValidationError::InconsistentLimits { .. })
        ));

        let no_recursion = RunLimits {
            max_depth: 0,
            max_subcalls: 1,
            ..RunLimits::default()
        };
        assert!(matches!(
            no_recursion.validate(),
            Err(ValidationError::InconsistentLimits { .. })
        ));
    }

    #[test]
    fn python_execution_requires_explicit_remote_exec() {
        let action = ModelAction::ExecutePython {
            code: "print('hello')".into(),
            input_artifact_ids: Vec::new(),
            timeout_ms: None,
        };

        assert_eq!(
            action.validate_for(RunCapabilities::default(), RunLimits::default(), 0),
            Err(ValidationError::CapabilityRequired("remote_exec"))
        );
    }

    #[test]
    fn internet_search_has_no_model_action() {
        let error = serde_json::from_str::<ModelAction>(
            r#"{"type":"internet_search","query":"do not call the internet"}"#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("unknown variant"));
    }

    #[test]
    fn model_id_lists_are_small_and_unique() {
        let duplicate = ModelAction::Subcall {
            prompt: "focused".into(),
            context_ids: vec!["context-1".into(), "context-1".into()],
            max_output_tokens: None,
        };
        assert!(matches!(
            duplicate.validate_for(
                RunCapabilities {
                    remote_context: true,
                    ..RunCapabilities::default()
                },
                RunLimits::default(),
                0,
            ),
            Err(ValidationError::DuplicateIdentifier {
                field: "context_ids"
            })
        ));

        let too_many = ModelAction::ExecutePython {
            code: "print('safe remote task')".into(),
            input_artifact_ids: (0..=MAX_ACTION_IDS)
                .map(|index| format!("artifact-{index}"))
                .collect(),
            timeout_ms: None,
        };
        assert!(matches!(
            too_many.validate_for(
                RunCapabilities {
                    remote_context: true,
                    remote_exec: true,
                    internet_search: false,
                },
                RunLimits::default(),
                0,
            ),
            Err(ValidationError::TooManyItems {
                field: "input_artifact_ids",
                maximum: MAX_ACTION_IDS,
                ..
            })
        ));
    }
}
