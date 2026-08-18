//! Durable, metadata-only state for RLM runs.
//!
//! This store deliberately accepts protocol JSON rather than arbitrary byte
//! blobs.  Large context, code outputs, and media belong in the remote
//! artifact store; SQLite keeps only bounded snapshots, events, and artifact
//! references.

use std::{
    fs::{self, OpenOptions},
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use chrono::{DateTime, Utc};
use fs2::FileExt;
use rlm_protocol::{RunEvent, RunEventKind, RunSnapshot, RunStatus, ValidationError};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use thiserror::Error;
use uuid::Uuid;

/// Event payloads are metadata, summaries, or references, never bulk output.
pub const MAX_EVENT_PAYLOAD_BYTES: usize = 64 * 1024;
/// A snapshot may contain the final answer, but remains intentionally small.
pub const MAX_RUN_SNAPSHOT_BYTES: usize = 256 * 1024;
pub const DEFAULT_EVENT_PAGE_SIZE: usize = 100;
pub const MAX_EVENT_PAGE_SIZE: usize = 512;

const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid run snapshot: {0}")]
    Validation(#[from] ValidationError),
    #[error("filesystem error: {0}")]
    Io(#[from] std::io::Error),
    #[error("run {0} was not found")]
    NotFound(Uuid),
    #[error("run {0} already exists")]
    AlreadyExists(Uuid),
    #[error("run status cannot transition from {from} to {to}")]
    InvalidTransition { from: RunStatus, to: RunStatus },
    #[error("event payload is {actual} bytes; maximum is {max}")]
    EventPayloadTooLarge { actual: usize, max: usize },
    #[error("run snapshot is {actual} bytes; maximum is {max}")]
    SnapshotTooLarge { actual: usize, max: usize },
    #[error("database schema version {found} is newer than supported version {supported}")]
    UnsupportedSchema { found: i64, supported: i64 },
    #[error("database contains invalid run status {0:?}")]
    InvalidStatus(String),
    #[error("database contains invalid millisecond timestamp {0}")]
    InvalidTimestamp(i64),
    #[error("numeric value {0} cannot be represented by SQLite")]
    NumericOverflow(u64),
    #[error("run store lock was poisoned")]
    LockPoisoned,
    #[error("database path must not be a symbolic link")]
    SymlinkDatabase,
    #[error("another rlmd process already owns this database")]
    AlreadyLocked,
}

/// A cheap-to-clone handle around the single SQLite writer connection.
///
/// SQLite is synchronous, so callers in async request paths should keep these
/// operations short (as designed) or invoke them through `spawn_blocking`.
#[derive(Clone)]
pub struct RunStore {
    connection: Arc<Mutex<Connection>>,
    _process_lock: Option<Arc<std::fs::File>>,
}

pub struct TransitionResult {
    pub snapshot: RunSnapshot,
    pub events: Vec<RunEvent>,
    pub changed: bool,
}

impl RunStore {
    /// Open (or create) the state database and apply forward-only migrations.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        let in_memory = path == Path::new(":memory:");
        let mut process_lock = None;
        if !in_memory {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    create_private_directory(parent)?;
                }
            }
            process_lock = Some(Arc::new(acquire_process_lock(path)?));
            prepare_private_database(path)?;
        }

        let mut connection = Connection::open(path)?;
        configure_connection(&connection)?;
        migrate(&mut connection)?;
        if !in_memory {
            secure_sqlite_files(path)?;
        }

        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
            _process_lock: process_lock,
        })
    }

    /// Insert a new run. Protocol event sequence ownership starts at zero.
    pub fn create_run(&self, snapshot: &RunSnapshot) -> Result<RunSnapshot, StoreError> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let run_id = snapshot.run_id;
        let exists = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM runs WHERE run_id = ?1)",
            [run_id.to_string()],
            |row| row.get::<_, bool>(0),
        )?;
        if exists {
            return Err(StoreError::AlreadyExists(run_id));
        }

        let mut canonical = snapshot.clone();
        canonical.next_sequence = 0;
        canonical.created_at = truncate_to_millis(canonical.created_at)?;
        canonical.updated_at = truncate_to_millis(canonical.updated_at)?;
        canonical.validate()?;
        let snapshot_json = encode_snapshot(&canonical)?;
        transaction.execute(
            "INSERT INTO runs (
                run_id, status, snapshot_json, created_at_ms, updated_at_ms,
                next_event_sequence, cancel_requested
             ) VALUES (?1, ?2, ?3, ?4, ?5, 0, 0)",
            params![
                run_id.to_string(),
                status_to_str(&canonical.status),
                snapshot_json,
                canonical.created_at.timestamp_millis(),
                canonical.updated_at.timestamp_millis(),
            ],
        )?;
        transaction.commit()?;
        Ok(canonical)
    }

    pub fn get_run(&self, run_id: Uuid) -> Result<Option<RunSnapshot>, StoreError> {
        let connection = self.lock()?;
        load_run(&connection, run_id).map(|row| row.map(|row| row.snapshot))
    }

    /// Mark runs left nonterminal by a previous process as failed. This is run
    /// once at startup before the HTTP listener accepts new work.
    pub fn fail_interrupted_runs(&self, reason: &str) -> Result<usize, StoreError> {
        let run_ids = {
            let connection = self.lock()?;
            let mut statement = connection.prepare(
                "SELECT run_id FROM runs WHERE status IN ('queued', 'running') ORDER BY created_at_ms",
            )?;
            let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
            rows.map(|row| {
                let value = row?;
                Uuid::parse_str(&value).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?
        };

        let mut recovered = 0;
        for run_id in run_ids {
            let snapshot_reason = reason.to_owned();
            let event_reason = reason.to_owned();
            let result = self.transition_run(
                run_id,
                RunStatus::Failed,
                move |snapshot| snapshot.error = Some(snapshot_reason),
                move |from, _| {
                    vec![
                        RunEventKind::StatusChanged {
                            from,
                            to: RunStatus::Failed,
                        },
                        RunEventKind::RunFailed {
                            error: event_reason,
                        },
                    ]
                },
            )?;
            recovered += usize::from(result.changed);
        }
        Ok(recovered)
    }

    /// Replace the mutable snapshot fields while preserving store-owned
    /// creation time and next event sequence.
    pub fn update_run(&self, snapshot: &RunSnapshot) -> Result<RunSnapshot, StoreError> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let stored = load_run(&transaction, snapshot.run_id)?
            .ok_or(StoreError::NotFound(snapshot.run_id))?;

        let mut canonical = snapshot.clone();
        canonical.created_at = stored.snapshot.created_at;
        canonical.updated_at = now_millis();
        canonical.next_sequence = stored.snapshot.next_sequence;
        let previous_status = stored.snapshot.status;
        if canonical.status != previous_status {
            return Err(StoreError::InvalidTransition {
                from: previous_status,
                to: canonical.status,
            });
        }
        if stored.cancel_requested && canonical.status != RunStatus::Cancelled {
            return Err(StoreError::InvalidTransition {
                from: RunStatus::Cancelled,
                to: canonical.status,
            });
        }
        canonical.validate()?;
        let snapshot_json = encode_snapshot(&canonical)?;
        transaction.execute(
            "UPDATE runs
             SET status = ?2, snapshot_json = ?3, updated_at_ms = ?4
             WHERE run_id = ?1",
            params![
                snapshot.run_id.to_string(),
                status_to_str(&canonical.status),
                snapshot_json,
                canonical.updated_at.timestamp_millis(),
            ],
        )?;
        transaction.commit()?;
        Ok(canonical)
    }

    /// Atomically change run status and append all events describing that
    /// transition. If another actor already made the run terminal, this is an
    /// idempotent no-op and returns the actual terminal snapshot.
    pub fn transition_run<Mutate, Events>(
        &self,
        run_id: Uuid,
        target: RunStatus,
        mutate: Mutate,
        events: Events,
    ) -> Result<TransitionResult, StoreError>
    where
        Mutate: FnOnce(&mut RunSnapshot),
        Events: FnOnce(RunStatus, &RunSnapshot) -> Vec<RunEventKind>,
    {
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let stored = load_run(&transaction, run_id)?.ok_or(StoreError::NotFound(run_id))?;
        let previous = stored.snapshot.status;

        if previous.is_terminal() || previous == target {
            transaction.commit()?;
            return Ok(TransitionResult {
                snapshot: stored.snapshot,
                events: Vec::new(),
                changed: false,
            });
        }
        if !previous.can_transition_to(target) {
            return Err(StoreError::InvalidTransition {
                from: previous,
                to: target,
            });
        }

        let mut snapshot = stored.snapshot;
        snapshot.status = target;
        mutate(&mut snapshot);
        snapshot.status = target;
        snapshot.updated_at = now_millis();
        let event_kinds = events(previous, &snapshot);
        let encoded_events = event_kinds
            .iter()
            .map(encode_event_payload)
            .collect::<Result<Vec<_>, _>>()?;
        let first_sequence = snapshot.next_sequence;
        let event_count =
            u64::try_from(event_kinds.len()).map_err(|_| StoreError::NumericOverflow(u64::MAX))?;
        snapshot.next_sequence = first_sequence
            .checked_add(event_count)
            .ok_or(StoreError::NumericOverflow(first_sequence))?;
        snapshot.validate()?;
        let snapshot_json = encode_snapshot(&snapshot)?;

        let mut committed_events = Vec::with_capacity(event_kinds.len());
        for (index, (kind, payload_json)) in event_kinds.into_iter().zip(encoded_events).enumerate()
        {
            let sequence = first_sequence
                .checked_add(index as u64)
                .ok_or(StoreError::NumericOverflow(first_sequence))?;
            transaction.execute(
                "INSERT INTO run_events (
                    run_id, sequence, occurred_at_ms, payload_json
                 ) VALUES (?1, ?2, ?3, ?4)",
                params![
                    run_id.to_string(),
                    to_sql_integer(sequence)?,
                    snapshot.updated_at.timestamp_millis(),
                    payload_json,
                ],
            )?;
            committed_events.push(RunEvent {
                run_id,
                sequence,
                occurred_at: snapshot.updated_at,
                kind,
            });
        }
        transaction.execute(
            "UPDATE runs
             SET status = ?2, snapshot_json = ?3, updated_at_ms = ?4,
                 next_event_sequence = ?5,
                 cancel_requested = CASE WHEN ?2 = 'cancelled' THEN 1 ELSE cancel_requested END
             WHERE run_id = ?1",
            params![
                run_id.to_string(),
                status_to_str(&snapshot.status),
                snapshot_json,
                snapshot.updated_at.timestamp_millis(),
                to_sql_integer(snapshot.next_sequence)?,
            ],
        )?;
        transaction.commit()?;
        Ok(TransitionResult {
            snapshot,
            events: committed_events,
            changed: true,
        })
    }

    pub fn is_cancel_requested(&self, run_id: Uuid) -> Result<bool, StoreError> {
        let connection = self.lock()?;
        connection
            .query_row(
                "SELECT cancel_requested FROM runs WHERE run_id = ?1",
                [run_id.to_string()],
                |row| row.get(0),
            )
            .optional()?
            .ok_or(StoreError::NotFound(run_id))
    }

    /// Append one typed event and allocate its sequence number atomically.
    pub fn append_event(&self, run_id: Uuid, kind: RunEventKind) -> Result<RunEvent, StoreError> {
        let payload_json = encode_event_payload(&kind)?;
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let stored = load_run(&transaction, run_id)?.ok_or(StoreError::NotFound(run_id))?;
        let sequence = stored.snapshot.next_sequence;
        let sequence_sql = to_sql_integer(sequence)?;
        let next_sequence = sequence
            .checked_add(1)
            .ok_or(StoreError::NumericOverflow(sequence))?;
        let next_sequence_sql = to_sql_integer(next_sequence)?;
        let occurred_at = now_millis();

        transaction.execute(
            "INSERT INTO run_events (
                run_id, sequence, occurred_at_ms, payload_json
             ) VALUES (?1, ?2, ?3, ?4)",
            params![
                run_id.to_string(),
                sequence_sql,
                occurred_at.timestamp_millis(),
                payload_json,
            ],
        )?;

        let mut snapshot = stored.snapshot;
        snapshot.next_sequence = next_sequence;
        snapshot.updated_at = occurred_at;
        let snapshot_json = encode_snapshot(&snapshot)?;
        transaction.execute(
            "UPDATE runs
             SET snapshot_json = ?2, updated_at_ms = ?3,
                 next_event_sequence = ?4
             WHERE run_id = ?1",
            params![
                run_id.to_string(),
                snapshot_json,
                occurred_at.timestamp_millis(),
                next_sequence_sql,
            ],
        )?;
        transaction.commit()?;

        Ok(RunEvent {
            run_id,
            sequence,
            occurred_at,
            kind,
        })
    }

    /// Return events after `after_sequence`, oldest first. Excessive limits are
    /// clamped so one client cannot materialize unbounded state in memory.
    pub fn list_events(
        &self,
        run_id: Uuid,
        after_sequence: Option<u64>,
        limit: usize,
    ) -> Result<Vec<RunEvent>, StoreError> {
        let connection = self.lock()?;
        let exists = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM runs WHERE run_id = ?1)",
            [run_id.to_string()],
            |row| row.get::<_, bool>(0),
        )?;
        if !exists {
            return Err(StoreError::NotFound(run_id));
        }

        // Protocol sequences start at zero, so -1 is the natural cursor before
        // the first event and remains private to this SQL query.
        let after_sequence = match after_sequence {
            Some(sequence) => to_sql_integer(sequence)?,
            None => -1,
        };
        let limit = limit.clamp(1, MAX_EVENT_PAGE_SIZE) as i64;
        let mut statement = connection.prepare_cached(
            "SELECT sequence, occurred_at_ms, payload_json
             FROM run_events
             WHERE run_id = ?1 AND sequence > ?2
             ORDER BY sequence ASC
             LIMIT ?3",
        )?;
        let mut rows = statement.query(params![run_id.to_string(), after_sequence, limit])?;
        let mut events = Vec::with_capacity(limit as usize);
        while let Some(row) = rows.next()? {
            let sequence_sql: i64 = row.get(0)?;
            let occurred_at_ms: i64 = row.get(1)?;
            let payload_json: String = row.get(2)?;
            events.push(RunEvent {
                run_id,
                sequence: u64::try_from(sequence_sql)
                    .map_err(|_| StoreError::NumericOverflow(sequence_sql as u64))?,
                occurred_at: timestamp_from_millis(occurred_at_ms)?,
                kind: serde_json::from_str(&payload_json)?,
            });
        }
        Ok(events)
    }

    fn lock(&self) -> Result<MutexGuard<'_, Connection>, StoreError> {
        self.connection.lock().map_err(|_| StoreError::LockPoisoned)
    }
}

fn appended_path(path: &Path, suffix: &str) -> std::path::PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    std::path::PathBuf::from(value)
}

#[cfg(unix)]
fn acquire_process_lock(path: &Path) -> Result<std::fs::File, StoreError> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let lock_path = appended_path(path, ".lock");
    if lock_path.exists() && fs::symlink_metadata(&lock_path)?.file_type().is_symlink() {
        return Err(StoreError::SymlinkDatabase);
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(&lock_path)?;
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o600))?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(file),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            Err(StoreError::AlreadyLocked)
        }
        Err(error) => Err(StoreError::Io(error)),
    }
}

#[cfg(not(unix))]
fn acquire_process_lock(path: &Path) -> Result<std::fs::File, StoreError> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(appended_path(path, ".lock"))?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(file),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            Err(StoreError::AlreadyLocked)
        }
        Err(error) => Err(StoreError::Io(error)),
    }
}

#[cfg(unix)]
fn create_private_directory(path: &Path) -> Result<(), StoreError> {
    use std::os::unix::fs::DirBuilderExt;

    if !path.exists() {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(path)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn create_private_directory(path: &Path) -> Result<(), StoreError> {
    fs::create_dir_all(path)?;
    Ok(())
}

#[cfg(unix)]
fn prepare_private_database(path: &Path) -> Result<(), StoreError> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    if path.exists() && fs::symlink_metadata(path)?.file_type().is_symlink() {
        return Err(StoreError::SymlinkDatabase);
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn prepare_private_database(_path: &Path) -> Result<(), StoreError> {
    Ok(())
}

#[cfg(unix)]
fn secure_sqlite_files(path: &Path) -> Result<(), StoreError> {
    use std::os::unix::fs::PermissionsExt;

    for candidate in [
        path.to_path_buf(),
        appended_path(path, "-wal"),
        appended_path(path, "-shm"),
        appended_path(path, ".lock"),
    ] {
        if candidate.exists() {
            fs::set_permissions(candidate, fs::Permissions::from_mode(0o600))?;
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn secure_sqlite_files(_path: &Path) -> Result<(), StoreError> {
    Ok(())
}

struct StoredRun {
    snapshot: RunSnapshot,
    cancel_requested: bool,
}

fn configure_connection(connection: &Connection) -> Result<(), StoreError> {
    connection.busy_timeout(Duration::from_secs(5))?;
    connection.pragma_update(None, "foreign_keys", true)?;
    connection.pragma_update(None, "journal_mode", "WAL")?;
    connection.pragma_update(None, "synchronous", "NORMAL")?;
    connection.pragma_update(None, "trusted_schema", false)?;
    connection.set_prepared_statement_cache_capacity(24);
    Ok(())
}

fn migrate(connection: &mut Connection) -> Result<(), StoreError> {
    let version: i64 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if version > SCHEMA_VERSION {
        return Err(StoreError::UnsupportedSchema {
            found: version,
            supported: SCHEMA_VERSION,
        });
    }
    if version == SCHEMA_VERSION {
        return Ok(());
    }

    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute_batch(
        "CREATE TABLE runs (
            run_id              TEXT PRIMARY KEY NOT NULL,
            status              TEXT NOT NULL CHECK (
                status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
            ),
            snapshot_json       TEXT NOT NULL CHECK (
                length(CAST(snapshot_json AS BLOB)) <= 262144
            ),
            created_at_ms       INTEGER NOT NULL,
            updated_at_ms       INTEGER NOT NULL,
            next_event_sequence INTEGER NOT NULL DEFAULT 0 CHECK (next_event_sequence >= 0),
            cancel_requested    INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1))
        ) STRICT;

        CREATE TABLE run_events (
            run_id         TEXT NOT NULL,
            sequence       INTEGER NOT NULL CHECK (sequence >= 0),
            occurred_at_ms INTEGER NOT NULL,
            payload_json   TEXT NOT NULL CHECK (
                length(CAST(payload_json AS BLOB)) <= 65536
            ),
            PRIMARY KEY (run_id, sequence),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        ) STRICT, WITHOUT ROWID;

        CREATE INDEX run_events_time
            ON run_events(run_id, occurred_at_ms);

        PRAGMA user_version = 1;",
    )?;
    transaction.commit()?;
    Ok(())
}

fn load_run(connection: &Connection, run_id: Uuid) -> Result<Option<StoredRun>, StoreError> {
    let row = connection
        .query_row(
            "SELECT status, snapshot_json, created_at_ms, updated_at_ms,
                    next_event_sequence, cancel_requested
             FROM runs WHERE run_id = ?1",
            [run_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, bool>(5)?,
                ))
            },
        )
        .optional()?;

    let Some((status, snapshot_json, created_at_ms, updated_at_ms, next_sequence, cancelled)) = row
    else {
        return Ok(None);
    };
    let mut snapshot: RunSnapshot = serde_json::from_str(&snapshot_json)?;
    snapshot.run_id = run_id;
    snapshot.status = status_from_str(&status)?;
    snapshot.created_at = timestamp_from_millis(created_at_ms)?;
    snapshot.updated_at = timestamp_from_millis(updated_at_ms)?;
    snapshot.next_sequence = u64::try_from(next_sequence)
        .map_err(|_| StoreError::NumericOverflow(next_sequence as u64))?;
    Ok(Some(StoredRun {
        snapshot,
        cancel_requested: cancelled,
    }))
}

fn encode_snapshot(snapshot: &RunSnapshot) -> Result<String, StoreError> {
    let bytes = serde_json::to_vec(snapshot)?;
    if bytes.len() > MAX_RUN_SNAPSHOT_BYTES {
        return Err(StoreError::SnapshotTooLarge {
            actual: bytes.len(),
            max: MAX_RUN_SNAPSHOT_BYTES,
        });
    }
    Ok(String::from_utf8(bytes).expect("serde_json always emits UTF-8"))
}

fn encode_event_payload(kind: &RunEventKind) -> Result<String, StoreError> {
    let bytes = serde_json::to_vec(kind)?;
    if bytes.len() > MAX_EVENT_PAYLOAD_BYTES {
        return Err(StoreError::EventPayloadTooLarge {
            actual: bytes.len(),
            max: MAX_EVENT_PAYLOAD_BYTES,
        });
    }
    Ok(String::from_utf8(bytes).expect("serde_json always emits UTF-8"))
}

fn timestamp_from_millis(value: i64) -> Result<DateTime<Utc>, StoreError> {
    DateTime::<Utc>::from_timestamp_millis(value).ok_or(StoreError::InvalidTimestamp(value))
}

fn truncate_to_millis(value: DateTime<Utc>) -> Result<DateTime<Utc>, StoreError> {
    timestamp_from_millis(value.timestamp_millis())
}

fn now_millis() -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp_millis(Utc::now().timestamp_millis())
        .expect("the current time is representable as Unix milliseconds")
}

fn to_sql_integer(value: u64) -> Result<i64, StoreError> {
    i64::try_from(value).map_err(|_| StoreError::NumericOverflow(value))
}

fn status_to_str(status: &RunStatus) -> &'static str {
    match status {
        RunStatus::Queued => "queued",
        RunStatus::Running => "running",
        RunStatus::Succeeded => "succeeded",
        RunStatus::Failed => "failed",
        RunStatus::Cancelled => "cancelled",
    }
}

fn status_from_str(status: &str) -> Result<RunStatus, StoreError> {
    match status {
        "queued" => Ok(RunStatus::Queued),
        "running" => Ok(RunStatus::Running),
        "succeeded" => Ok(RunStatus::Succeeded),
        "failed" => Ok(RunStatus::Failed),
        "cancelled" => Ok(RunStatus::Cancelled),
        other => Err(StoreError::InvalidStatus(other.to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use rlm_protocol::Usage;
    use tempfile::TempDir;

    use super::*;

    fn snapshot(run_id: Uuid) -> RunSnapshot {
        let now = Utc::now();
        RunSnapshot {
            run_id,
            status: RunStatus::Queued,
            next_sequence: 99, // create_run canonicalizes store-owned sequencing.
            created_at: now,
            updated_at: now,
            usage: Usage::default(),
            final_answer: None,
            error: None,
            artifacts: Vec::new(),
        }
    }

    fn store() -> (TempDir, RunStore) {
        let directory = tempfile::tempdir().unwrap();
        let store = RunStore::open(directory.path().join("state.sqlite3")).unwrap();
        (directory, store)
    }

    #[test]
    fn creates_updates_and_reopens_a_run() {
        let (directory, store) = store();
        let run_id = Uuid::new_v4();
        let created = store.create_run(&snapshot(run_id)).unwrap();
        assert_eq!(created.next_sequence, 0);
        assert!(matches!(created.status, RunStatus::Queued));
        assert!(matches!(
            store.create_run(&snapshot(run_id)),
            Err(StoreError::AlreadyExists(id)) if id == run_id
        ));

        let transition = store
            .transition_run(run_id, RunStatus::Running, |_| {}, |_, _| Vec::new())
            .unwrap();
        let mut update = transition.snapshot;
        update.next_sequence = 400;
        let updated = store.update_run(&update).unwrap();
        assert!(matches!(updated.status, RunStatus::Running));
        assert_eq!(updated.next_sequence, 0);
        drop(store);

        let reopened = RunStore::open(directory.path().join("state.sqlite3")).unwrap();
        let loaded = reopened.get_run(run_id).unwrap().unwrap();
        assert!(matches!(loaded.status, RunStatus::Running));
        assert_eq!(loaded.next_sequence, 0);
    }

    #[test]
    fn appends_monotonic_sequences_across_threads() {
        let (_directory, store) = store();
        let run_id = Uuid::new_v4();
        store.create_run(&snapshot(run_id)).unwrap();

        let mut threads = Vec::new();
        for _ in 0..4 {
            let store = store.clone();
            threads.push(thread::spawn(move || {
                for _ in 0..12 {
                    store.append_event(run_id, RunEventKind::RunQueued).unwrap();
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }

        let events = store.list_events(run_id, None, usize::MAX).unwrap();
        assert_eq!(events.len(), 48);
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            (0..48).collect::<Vec<_>>()
        );
        let page = store.list_events(run_id, Some(39), 2).unwrap();
        assert_eq!(
            page.iter().map(|event| event.sequence).collect::<Vec<_>>(),
            vec![40, 41]
        );
        assert_eq!(store.list_events(run_id, Some(0), 0).unwrap().len(), 1);
        assert_eq!(store.get_run(run_id).unwrap().unwrap().next_sequence, 48);
    }

    #[test]
    fn cancellation_is_idempotent() {
        let (_directory, store) = store();
        let run_id = Uuid::new_v4();
        store.create_run(&snapshot(run_id)).unwrap();

        let first = store
            .transition_run(run_id, RunStatus::Cancelled, |_| {}, |_, _| Vec::new())
            .unwrap()
            .snapshot;
        let second = store
            .transition_run(run_id, RunStatus::Cancelled, |_| {}, |_, _| Vec::new())
            .unwrap()
            .snapshot;
        assert!(matches!(first.status, RunStatus::Cancelled));
        assert!(matches!(second.status, RunStatus::Cancelled));
        assert_eq!(first.updated_at, second.updated_at);
        assert!(store.is_cancel_requested(run_id).unwrap());

        let mut stale_update = second;
        stale_update.status = RunStatus::Running;
        assert!(matches!(
            store.update_run(&stale_update),
            Err(StoreError::InvalidTransition {
                from: RunStatus::Cancelled,
                to: RunStatus::Running,
            })
        ));
    }

    #[test]
    fn rejects_bulk_event_payloads_without_consuming_a_sequence() {
        let (_directory, store) = store();
        let run_id = Uuid::new_v4();
        store.create_run(&snapshot(run_id)).unwrap();
        let result = store.append_event(
            run_id,
            RunEventKind::RunFailed {
                error: "x".repeat(MAX_EVENT_PAYLOAD_BYTES),
            },
        );
        assert!(matches!(
            result,
            Err(StoreError::EventPayloadTooLarge { .. })
        ));

        let event = store.append_event(run_id, RunEventKind::RunQueued).unwrap();
        assert_eq!(event.sequence, 0);
    }

    #[test]
    fn missing_runs_are_reported() {
        let (_directory, store) = store();
        let run_id = Uuid::new_v4();
        assert!(matches!(
            store.transition_run(run_id, RunStatus::Cancelled, |_| {}, |_, _| Vec::new()),
            Err(StoreError::NotFound(id)) if id == run_id
        ));
        assert!(matches!(
            store.list_events(run_id, None, DEFAULT_EVENT_PAGE_SIZE),
            Err(StoreError::NotFound(id)) if id == run_id
        ));
    }

    #[test]
    fn enables_wal_and_migrates_once() {
        let (_directory, store) = store();
        let connection = store.lock().unwrap();
        let journal: String = connection
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        let version: i64 = connection
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(journal, "wal");
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[cfg(unix)]
    #[test]
    fn creates_private_state_files_and_rejects_symlinks() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir().unwrap();
        let private_dir = directory.path().join("private-state");
        let database = private_dir.join("state.sqlite3");
        let store = RunStore::open(&database).unwrap();
        store.create_run(&snapshot(Uuid::new_v4())).unwrap();

        assert_eq!(
            fs::metadata(&private_dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&database).unwrap().permissions().mode() & 0o777,
            0o600
        );
        for suffix in ["-wal", "-shm"] {
            let sidecar = std::path::PathBuf::from(format!("{}{suffix}", database.display()));
            if sidecar.exists() {
                assert_eq!(
                    fs::metadata(sidecar).unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
        }

        let target = directory.path().join("target.sqlite3");
        fs::write(&target, []).unwrap();
        let link = directory.path().join("linked.sqlite3");
        symlink(&target, &link).unwrap();
        assert!(matches!(
            RunStore::open(link),
            Err(StoreError::SymlinkDatabase)
        ));
    }

    #[test]
    fn startup_recovery_fails_nonterminal_runs_with_events() {
        let (_directory, store) = store();
        let queued_id = Uuid::new_v4();
        let running_id = Uuid::new_v4();
        store.create_run(&snapshot(queued_id)).unwrap();
        store.create_run(&snapshot(running_id)).unwrap();
        store
            .transition_run(running_id, RunStatus::Running, |_| {}, |_, _| Vec::new())
            .unwrap();

        assert_eq!(store.fail_interrupted_runs("process restarted").unwrap(), 2);
        assert_eq!(store.fail_interrupted_runs("again").unwrap(), 0);
        for run_id in [queued_id, running_id] {
            let recovered = store.get_run(run_id).unwrap().unwrap();
            assert_eq!(recovered.status, RunStatus::Failed);
            assert_eq!(recovered.error.as_deref(), Some("process restarted"));
            let events = store.list_events(run_id, None, 10).unwrap();
            assert!(matches!(
                events.as_slice(),
                [
                    RunEvent {
                        kind: RunEventKind::StatusChanged { .. },
                        ..
                    },
                    RunEvent {
                        kind: RunEventKind::RunFailed { .. },
                        ..
                    }
                ]
            ));
        }
    }

    #[test]
    fn database_has_one_process_owner_but_memory_stores_are_independent() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("owned.sqlite3");
        let first = RunStore::open(&path).unwrap();
        assert!(matches!(
            RunStore::open(&path),
            Err(StoreError::AlreadyLocked)
        ));
        drop(first);
        RunStore::open(&path).unwrap();

        let memory_a = RunStore::open(":memory:").unwrap();
        let memory_b = RunStore::open(":memory:").unwrap();
        assert!(memory_a.get_run(Uuid::new_v4()).unwrap().is_none());
        assert!(memory_b.get_run(Uuid::new_v4()).unwrap().is_none());
        assert!(!std::path::Path::new(":memory:").exists());
    }
}
