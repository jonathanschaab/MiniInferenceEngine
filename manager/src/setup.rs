use axum::{
    Router,
    routing::{delete, get, post},
};
use std::sync::{Arc, Mutex};
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

use crate::{
    AppConfig, AppState, LogReloadHandle, SharedLogBuffer, append_chat_message, auth,
    clear_console_logs, delete_chat_session, delete_model, get_chat_session, get_console_loglevel,
    get_console_logs, get_download_progress, get_models, get_stats_data, get_status,
    handle_generate, list_chat_sessions, save_chat_session, serve_chat_js, serve_common_css,
    serve_common_js, serve_console_js, serve_console_ui, serve_memory_js, serve_memory_ui,
    serve_models_js, serve_models_ui, serve_settings_js, serve_settings_ui, serve_stats_js,
    serve_stats_ui, serve_ui, set_console_loglevel, trigger_benchmark, trigger_download,
    pause_download, truncate_chat_messages,
};

pub async fn init_db(
    config: &AppConfig,
) -> Result<surrealdb::Surreal<surrealdb::engine::any::Any>, Box<dyn std::error::Error>> {
    let jwt = match std::fs::read_to_string(&config.database.jwt_file_path) {
        Ok(content) => content,
        Err(e) => {
            let msg = format!(
                "Failed to read database JWT file '{}': {}",
                config.database.jwt_file_path, e
            );
            error!("{}", msg);
            return Err(msg.into());
        }
    };

    let mut db_client_opt = None;
    for attempt in 1..=3 {
        match surrealdb::engine::any::connect(&config.database.url).await {
            Ok(client) => {
                if let Err(e) = client.authenticate(jwt.trim()).await {
                    let msg = format!("SurrealDB authentication failed: {}", e);
                    error!("{}", msg);
                    return Err(msg.into());
                } else if let Err(e) = client
                    .use_ns(&config.database.namespace)
                    .use_db(&config.database.database)
                    .await
                {
                    let msg = format!("Failed to switch to namespace/database: {}", e);
                    error!("{}", msg);
                    return Err(msg.into());
                } else {
                    db_client_opt = Some(client);
                    break;
                }
            }
            Err(e) => {
                error!(
                    "Failed to connect to SurrealDB on attempt {}: {}",
                    attempt, e
                );
            }
        }
        if attempt < 3 {
            info!("Retrying database connection in 2 seconds...");
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    let db_client = match db_client_opt {
        Some(client) => client,
        None => {
            let msg =
                "Database is temporarily unavailable after 3 attempts. Gracefully shutting down.";
            error!("{}", msg);
            return Err(msg.into());
        }
    };

    let index_queries = "
        DEFINE INDEX IF NOT EXISTS chat_sessions_email_idx ON TABLE chat_sessions COLUMNS email;
        DEFINE INDEX IF NOT EXISTS chat_messages_session_idx ON TABLE chat_messages COLUMNS session_id, message_index;
        DEFINE INDEX IF NOT EXISTS telemetry_loads_timestamp_idx ON TABLE telemetry_loads COLUMNS timestamp;
        DEFINE INDEX IF NOT EXISTS telemetry_generations_timestamp_idx ON TABLE telemetry_generations COLUMNS timestamp;
    ";

    if let Err(e) = db_client.query(index_queries).await {
        let msg = format!("Failed to define database indexes: {}", e);
        error!("{}", msg);
        return Err(msg.into());
    }

    Ok(db_client)
}

pub fn init_logging(
    config: &AppConfig,
) -> (
    SharedLogBuffer,
    LogReloadHandle,
    tracing_appender::non_blocking::WorkerGuard,
) {
    // --- 1. CONSOLE LAYER ---
    let console_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(EnvFilter::new(&config.log_level_console));

    // --- 2. FILE LAYER ---
    let file_appender = tracing_appender::rolling::never(".", &config.log_file_name);
    // Bind the file_guard to keep the background writer active for the life of main()
    let (file_writer, file_guard) = tracing_appender::non_blocking(file_appender);
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(file_writer)
        .with_ansi(false) // Do not write ANSI color codes to the log file!
        .with_filter(EnvFilter::new(&config.log_level_file));

    // --- 3. IN-MEMORY BUFFER LAYER (RELOADABLE) ---
    let memory_buffer =
        SharedLogBuffer(Arc::new(Mutex::new((0, std::collections::VecDeque::new()))));
    let memory_filter = EnvFilter::new(&config.log_level_memory);
    let (reloadable_memory_filter, log_reload_handle) =
        tracing_subscriber::reload::Layer::new(memory_filter);
    let memory_layer = tracing_subscriber::fmt::layer()
        .with_writer(memory_buffer.clone())
        .with_ansi(false) // Send clean strings to the web UI
        .with_filter(reloadable_memory_filter);

    // Apply all registered layers
    tracing_subscriber::registry()
        .with(memory_layer)
        .with(file_layer)
        .with(console_layer)
        .init();

    (memory_buffer, log_reload_handle, file_guard)
}

pub fn build_web_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Public OAuth
        .route("/auth/login", get(auth::login_handler))
        .route("/auth/google/callback", get(auth::callback_handler))
        .route("/auth/logout", get(auth::logout_handler))
        // Protected UIs (They redirect if session is missing)
        .route("/", get(serve_ui))
        .route("/settings", get(serve_settings_ui))
        .route("/models", get(serve_models_ui))
        .route("/stats", get(serve_stats_ui))
        .route("/memory", get(serve_memory_ui))
        .route("/console", get(serve_console_ui))
        .route("/js/chat.js", get(serve_chat_js))
        .route("/js/models.js", get(serve_models_js))
        .route("/js/stats.js", get(serve_stats_js))
        .route("/js/settings.js", get(serve_settings_js))
        .route("/js/memory.js", get(serve_memory_js))
        .route("/js/console.js", get(serve_console_js))
        .route("/js/common.js", get(serve_common_js))
        .route("/css/common.css", get(serve_common_css))
        // Settings APIs (They check session manually)
        .route(
            "/api/settings/keys",
            get(auth::list_keys_handler).post(auth::create_key_handler),
        )
        .route(
            "/api/settings/keys/{hash}",
            delete(auth::delete_key_handler),
        )
}

pub fn build_engine_api_routes(state: Arc<AppState>) -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/generate", post(handle_generate))
        .route("/api/stats/collect", post(trigger_benchmark))
        .route(
            "/api/chat/sessions",
            get(list_chat_sessions).post(save_chat_session),
        )
        .route(
            "/api/chat/sessions/{id}",
            get(get_chat_session).delete(delete_chat_session),
        )
        .route(
            "/api/chat/sessions/{id}/messages",
            post(append_chat_message),
        )
        .route(
            "/api/chat/sessions/{id}/messages/{index}",
            delete(truncate_chat_messages),
        )
        .route("/api/models", get(get_models))
        .route("/api/models/download/progress", get(get_download_progress))
        .route(
            "/api/models/{id}/download",
            post(trigger_download).delete(delete_model),
        )
        .route(
            "/api/models/{id}/pause",
            post(pause_download),
        )
        .route("/api/status", get(get_status))
        .route("/api/stats/data", get(get_stats_data))
        .route(
            "/api/console/logs",
            get(get_console_logs).delete(clear_console_logs),
        )
        .route(
            "/api/console/loglevel",
            get(get_console_loglevel).post(set_console_loglevel),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            state,
            auth::dual_auth_middleware,
        ))
}
