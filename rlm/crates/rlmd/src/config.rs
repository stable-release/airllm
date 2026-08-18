use std::{net::SocketAddr, path::PathBuf, time::Duration};

use clap::Parser;
use thiserror::Error;
use url::Url;

const DEFAULT_BIND: &str = "127.0.0.1:8090";
const DEFAULT_MODEL_URL: &str = "http://127.0.0.1:8080";

/// Offline-first RLM control plane for a local MLX model.
#[derive(Clone, Debug, Parser)]
#[command(version, about)]
pub struct Config {
    /// Loopback address exposed by rlmd. Use a TLS proxy/Tailscale for remote access.
    #[arg(long, env = "RLMD_BIND", default_value = DEFAULT_BIND)]
    pub bind: SocketAddr,

    /// Loopback-only mlx_lm.server base URL.
    #[arg(long, env = "RLMD_MODEL_URL", default_value = DEFAULT_MODEL_URL)]
    pub model_url: Url,

    /// Small SQLite state database. Bulk context and media never belong here.
    #[arg(long, env = "RLMD_DATABASE", default_value = "rlmd.db")]
    pub database: PathBuf,

    /// Bearer token for the public rlmd API.
    #[arg(long, env = "RLMD_API_TOKEN")]
    pub api_token: Option<String>,

    /// Explicit remote worker base URL. Without this, context/code actions fail closed.
    #[arg(long, env = "RLMD_WORKER_URL")]
    pub worker_url: Option<Url>,

    /// Bearer token used only for the configured remote worker.
    #[arg(long, env = "RLMD_WORKER_TOKEN")]
    pub worker_token: Option<String>,

    /// Hard timeout for one model completion.
    #[arg(long, env = "RLMD_MODEL_TIMEOUT_SECS", default_value_t = 120)]
    pub model_timeout_secs: u64,

    /// Hard timeout for one remote worker task.
    #[arg(long, env = "RLMD_WORKER_TIMEOUT_SECS", default_value_t = 30)]
    pub worker_timeout_secs: u64,

    /// Maximum accepted request size before JSON decoding.
    #[arg(long, env = "RLMD_REQUEST_BYTES", default_value_t = 1_048_576)]
    pub request_bytes: usize,

    /// Maximum simultaneous runs. MLX calls remain serialized independently.
    #[arg(long, env = "RLMD_MAX_RUNS", default_value_t = 16)]
    pub max_runs: usize,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("RLMD_MODEL_URL must use http or https")]
    ModelScheme,
    #[error("RLMD_MODEL_URL must point to a loopback host")]
    ModelNotLoopback,
    #[error("RLMD_BIND must be loopback; expose rlmd through a trusted TLS proxy")]
    NonLoopbackBind,
    #[error("configured bearer tokens must contain at least 32 bytes")]
    WeakToken,
    #[error("RLMD_WORKER_TOKEN requires RLMD_WORKER_URL")]
    WorkerTokenWithoutUrl,
    #[error("invalid RLMD_WORKER_URL: {0}")]
    InvalidWorkerEndpoint(String),
    #[error("a non-loopback RLMD_WORKER_URL requires RLMD_WORKER_TOKEN")]
    RemoteWorkerWithoutToken,
    #[error("{0} must be greater than zero")]
    Zero(&'static str),
}

impl Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_loopback_model_url(&self.model_url)?;
        if !self.bind.ip().is_loopback() {
            return Err(ConfigError::NonLoopbackBind);
        }
        if self
            .api_token
            .as_deref()
            .is_some_and(|token| token.len() < 32)
            || self
                .worker_token
                .as_deref()
                .is_some_and(|token| token.len() < 32)
        {
            return Err(ConfigError::WeakToken);
        }
        if self.worker_url.is_none() && self.worker_token.is_some() {
            return Err(ConfigError::WorkerTokenWithoutUrl);
        }
        if let Some(url) = &self.worker_url {
            crate::executor::validate_worker_url(url)
                .map_err(|error| ConfigError::InvalidWorkerEndpoint(error.to_string()))?;
            let loopback = url.host().is_some_and(|host| match host {
                url::Host::Ipv4(address) => address.is_loopback(),
                url::Host::Ipv6(address) => address.is_loopback(),
                url::Host::Domain(_) => false,
            });
            if !loopback && self.worker_token.is_none() {
                return Err(ConfigError::RemoteWorkerWithoutToken);
            }
        }
        for (name, value) in [
            ("RLMD_MODEL_TIMEOUT_SECS", self.model_timeout_secs as usize),
            (
                "RLMD_WORKER_TIMEOUT_SECS",
                self.worker_timeout_secs as usize,
            ),
            ("RLMD_REQUEST_BYTES", self.request_bytes),
            ("RLMD_MAX_RUNS", self.max_runs),
        ] {
            if value == 0 {
                return Err(ConfigError::Zero(name));
            }
        }
        Ok(())
    }

    pub fn model_timeout(&self) -> Duration {
        Duration::from_secs(self.model_timeout_secs)
    }

    pub fn worker_timeout(&self) -> Duration {
        Duration::from_secs(self.worker_timeout_secs)
    }
}

pub fn validate_loopback_model_url(url: &Url) -> Result<(), ConfigError> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(ConfigError::ModelScheme);
    }
    let is_loopback = url.host().is_some_and(|host| match host {
        url::Host::Ipv4(address) => address.is_loopback(),
        url::Host::Ipv6(address) => address.is_loopback(),
        url::Host::Domain(_) => false,
    });
    if !is_loopback {
        return Err(ConfigError::ModelNotLoopback);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Config {
        Config::parse_from(["rlmd"])
    }

    #[test]
    fn defaults_are_offline_and_loopback_only() {
        let config = base();
        config.validate().unwrap();
        assert!(config.bind.ip().is_loopback());
        assert_eq!(config.model_url.as_str(), "http://127.0.0.1:8080/");
        assert!(config.worker_url.is_none());
    }

    #[test]
    fn rejects_remote_model_endpoint() {
        let mut config = base();
        config.model_url = Url::parse("https://api.example.com").unwrap();
        assert!(matches!(
            config.validate(),
            Err(ConfigError::ModelNotLoopback)
        ));
    }

    #[test]
    fn rejects_plaintext_public_bind() {
        let mut config = base();
        config.bind = "0.0.0.0:8090".parse().unwrap();
        assert!(matches!(
            config.validate(),
            Err(ConfigError::NonLoopbackBind)
        ));
        config.api_token = Some("a-real-secret-with-at-least-32-bytes".into());
        assert!(matches!(
            config.validate(),
            Err(ConfigError::NonLoopbackBind)
        ));
    }

    #[test]
    fn remote_worker_requires_tls() {
        let mut config = base();
        config.worker_url = Some(Url::parse("http://10.0.0.8:9090").unwrap());
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidWorkerEndpoint(_))
        ));
    }

    #[test]
    fn remote_worker_requires_authentication() {
        let mut config = base();
        config.worker_url = Some(Url::parse("https://worker.example.test").unwrap());
        assert!(matches!(
            config.validate(),
            Err(ConfigError::RemoteWorkerWithoutToken)
        ));
        config.worker_token = Some("a-real-secret-with-at-least-32-bytes".into());
        config.validate().unwrap();
    }
}
