use std::{error::Error, sync::Arc};

use clap::Parser;
use rlmd::{
    config::Config,
    engine::RlmEngine,
    executor::{RemoteWorkerClient, WorkerBackend},
    model::{MlxModelClient, ModelBackend},
    server::{AppState, router},
    store::RunStore,
};
use tokio::{net::TcpListener, sync::Semaphore};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::parse();
    config.validate()?;
    init_tracing();

    let store = RunStore::open(&config.database)?;
    let recovered = store.fail_interrupted_runs("rlmd restarted before the run completed")?;
    if recovered > 0 {
        warn!(recovered, "marked interrupted runs as failed");
    }
    let generation_slots = Arc::new(Semaphore::new(1));
    let model: Arc<dyn ModelBackend> = Arc::new(MlxModelClient::with_semaphore(
        config.model_url.clone(),
        config.model_timeout(),
        generation_slots,
    )?);
    let worker: Option<Arc<dyn WorkerBackend>> = config
        .worker_url
        .clone()
        .map(|url| {
            RemoteWorkerClient::new(url, config.worker_token.clone(), config.worker_timeout())
                .map(|client| Arc::new(client) as Arc<dyn WorkerBackend>)
        })
        .transpose()?;
    let engine = Arc::new(RlmEngine::new(model, worker, store, config.max_runs)?);

    if let Err(error) = engine.model_readiness().await {
        warn!(%error, "local MLX endpoint is not ready; rlmd will start but /readyz will fail");
    }

    let app = router(
        AppState::new(Arc::clone(&engine), config.api_token.clone()),
        config.request_bytes,
    );
    let listener = TcpListener::bind(config.bind).await?;
    info!(
        bind = %config.bind,
        model = %config.model_url,
        worker_configured = engine.worker_available(),
        "rlmd started in offline inference mode"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("rlmd=info,tower_http=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .compact()
        .init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("shutdown requested");
}
