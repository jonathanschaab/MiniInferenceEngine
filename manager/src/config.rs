use serde::{Deserialize, Serialize};
use tracing::{error, warn};

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct DatabaseConfig {
    pub url: String,
    pub jwt_file_path: String,
    pub namespace: String,
    pub database: String,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "ws://localhost:8001".to_string(),
            jwt_file_path: "database.jwt".to_string(),
            namespace: "mini_inference_engine".to_string(),
            database: "main".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub bind_address: String,
    pub oauth_redirect_uri: String,
    #[serde(default = "default_oauth_client_secret_path")]
    pub oauth_client_secret_path: String,
    #[serde(default = "default_oauth_auth_url")]
    pub oauth_auth_url: String,
    #[serde(default = "default_oauth_token_url")]
    pub oauth_token_url: String,
    #[serde(default = "default_oauth_userinfo_url")]
    pub oauth_userinfo_url: String,
    pub admin_emails: Vec<String>,
    pub user_emails: Vec<String>,
    pub secure_cookies: bool,
    #[serde(default)]
    pub gpu_device_index: u32,
    #[serde(default = "default_max_concurrent_downloads")]
    pub max_concurrent_downloads: usize,
    #[serde(default = "default_max_concurrent_chunk_downloads")]
    pub max_concurrent_chunk_downloads: usize,
    #[serde(default = "default_telemetry_retention_days")]
    pub telemetry_retention_days: u64,
    #[serde(default = "default_temp_file_retention_days")]
    pub temp_file_retention_days: u64,
    #[serde(default = "default_download_retry_max_attempts")]
    pub download_retry_max_attempts: u32,
    #[serde(default = "default_download_retry_backoff_seconds")]
    pub download_retry_backoff_seconds: u64,
    #[serde(default = "default_download_stream_chunk_timeout_seconds")]
    pub download_stream_chunk_timeout_seconds: u64,
    #[serde(default = "default_registry_connect_timeout_seconds")]
    pub registry_connect_timeout_seconds: u64,
    #[serde(default = "default_registry_request_timeout_seconds")]
    pub registry_request_timeout_seconds: u64,
    #[serde(default = "default_log_level_console")]
    pub log_level_console: String,
    #[serde(default = "default_log_level_file")]
    pub log_level_file: String,
    #[serde(default = "default_log_level_memory")]
    pub log_level_memory: String,
    #[serde(default = "default_log_file_name")]
    pub log_file_name: String,
    #[serde(default = "default_downloads_directory")]
    pub downloads_directory: String,
    #[serde(default = "default_hf_base_url")]
    pub hf_base_url: String,
    #[serde(default)]
    pub database: DatabaseConfig,
}

pub fn default_log_level_console() -> String {
    "info".to_string()
}
pub fn default_log_level_file() -> String {
    "warn".to_string()
}
pub fn default_log_level_memory() -> String {
    "debug".to_string()
}
pub fn default_log_file_name() -> String {
    "server.log".to_string()
}
fn default_oauth_client_secret_path() -> String {
    "client_secret.apps.googleusercontent.com.json".to_string()
}
fn default_oauth_auth_url() -> String {
    "https://accounts.google.com/o/oauth2/v2/auth".to_string()
}
fn default_oauth_token_url() -> String {
    "https://oauth2.googleapis.com/token".to_string()
}
fn default_oauth_userinfo_url() -> String {
    "https://www.googleapis.com/oauth2/v2/userinfo".to_string()
}
fn default_max_concurrent_downloads() -> usize {
    2
}
fn default_max_concurrent_chunk_downloads() -> usize {
    8
}
fn default_telemetry_retention_days() -> u64 {
    30
}
fn default_temp_file_retention_days() -> u64 {
    3
}
fn default_download_retry_max_attempts() -> u32 {
    5
}
fn default_download_retry_backoff_seconds() -> u64 {
    2
}
fn default_download_stream_chunk_timeout_seconds() -> u64 {
    30
}
fn default_registry_connect_timeout_seconds() -> u64 {
    5
}
fn default_registry_request_timeout_seconds() -> u64 {
    15
}

fn default_downloads_directory() -> String {
    "downloads".to_string()
}
fn default_hf_base_url() -> String {
    "https://huggingface.co".to_string()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:3000".to_string(), // Secure local default
            oauth_redirect_uri: "http://localhost:3000/auth/google/callback".to_string(),
            oauth_client_secret_path: default_oauth_client_secret_path(),
            oauth_auth_url: default_oauth_auth_url(),
            oauth_token_url: default_oauth_token_url(),
            oauth_userinfo_url: default_oauth_userinfo_url(),
            admin_emails: vec![],
            user_emails: vec![],
            secure_cookies: true,
            gpu_device_index: 0,
            max_concurrent_downloads: default_max_concurrent_downloads(),
            max_concurrent_chunk_downloads: default_max_concurrent_chunk_downloads(),
            telemetry_retention_days: default_telemetry_retention_days(),
            temp_file_retention_days: default_temp_file_retention_days(),
            download_retry_max_attempts: default_download_retry_max_attempts(),
            download_retry_backoff_seconds: default_download_retry_backoff_seconds(),
            download_stream_chunk_timeout_seconds: default_download_stream_chunk_timeout_seconds(),
            registry_connect_timeout_seconds: default_registry_connect_timeout_seconds(),
            registry_request_timeout_seconds: default_registry_request_timeout_seconds(),
            log_level_console: default_log_level_console(),
            log_level_file: default_log_level_file(),
            log_level_memory: default_log_level_memory(),
            log_file_name: default_log_file_name(),
            downloads_directory: default_downloads_directory(),
            hf_base_url: default_hf_base_url(),
            database: DatabaseConfig::default(),
        }
    }
}

impl AppConfig {
    /// Validates and clamps configuration values to safe operational limits.
    pub fn sanitize(&mut self) {
        if self.max_concurrent_downloads == 0 {
            warn!("max_concurrent_downloads cannot be 0. Enforcing minimum value of 1.");
            self.max_concurrent_downloads = 1;
        }
        if self.max_concurrent_chunk_downloads == 0 {
            warn!("max_concurrent_chunk_downloads cannot be 0. Enforcing minimum value of 1.");
            self.max_concurrent_chunk_downloads = 1;
        }
        if self.download_stream_chunk_timeout_seconds == 0 {
            warn!(
                "download_stream_chunk_timeout_seconds cannot be 0. Enforcing default value of 30 seconds."
            );
            self.download_stream_chunk_timeout_seconds =
                default_download_stream_chunk_timeout_seconds();
        }
    }

    pub async fn load() -> Self {
        let toml_path = crate::types::resolve_absolute_path("config.toml");
        let json_path = crate::types::resolve_absolute_path("config.json");
        let (mut config, needs_save) = if let Ok(data) = tokio::fs::read_to_string(&toml_path).await
        {
            match toml::from_str(&data) {
                Ok(c) => (c, false),
                Err(e) => {
                    error!(
                        "CRITICAL: Failed to parse config.toml: {}. Please fix the syntax errors. Falling back to defaults.",
                        e
                    );
                    (Self::default(), false) // Don't overwrite the user's broken file
                }
            }
        } else if let Ok(data) = tokio::fs::read_to_string(&json_path).await {
            match serde_json::from_str(&data) {
                Ok(c) => (c, true), // Fallback for backwards compatibility, save as TOML going forward
                Err(e) => {
                    error!(
                        "CRITICAL: Failed to parse config.json: {}. Please fix the syntax errors. Falling back to defaults.",
                        e
                    );
                    (Self::default(), false) // Don't overwrite the user's broken file
                }
            }
        } else {
            (Self::default(), true)
        };

        config.sanitize();

        if needs_save {
            match toml::to_string_pretty(&config) {
                Ok(toml_str) => {
                    if let Err(e) = tokio::fs::write(&toml_path, toml_str).await {
                        warn!("Failed to write config.toml: {}", e);
                    }
                }
                Err(e) => warn!("Failed to serialize configuration to TOML: {}", e),
            }
        }

        config
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
#[allow(clippy::indexing_slicing)]
#[allow(clippy::panic)]
#[allow(clippy::unreachable)]
#[allow(clippy::todo)]
#[allow(clippy::unimplemented)]
mod tests {
    use super::*;

    #[test]
    fn test_app_config_sanitize() {
        let mut config = AppConfig {
            max_concurrent_downloads: 0,
            max_concurrent_chunk_downloads: 0,
            download_stream_chunk_timeout_seconds: 0,
            ..Default::default()
        };

        config.sanitize();

        assert_eq!(config.max_concurrent_downloads, 1);
        assert_eq!(config.max_concurrent_chunk_downloads, 1);
        assert_eq!(
            config.download_stream_chunk_timeout_seconds,
            default_download_stream_chunk_timeout_seconds()
        );
    }
}
