use crate::AppState;
use axum::{
    Json,
    extract::{FromRequestParts, Path, Query, Request, State},
    http::{StatusCode, request::Parts},
    middleware::Next,
    response::{IntoResponse, Redirect, Response},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use oauth2::{
    AuthUrl, AuthorizationCode, ClientId, ClientSecret, CsrfToken, RedirectUrl, Scope,
    TokenResponse, TokenUrl, basic::BasicClient, reqwest::async_http_client,
};
use rand::{RngCore, thread_rng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, sync::Arc};
use surrealdb::{Surreal, engine::any::Any};
use tokio::sync::mpsc::UnboundedSender;
use tower_sessions::Session;
use tracing::error;

#[derive(Clone, Debug)]
pub struct CurrentUser {
    pub email: String,
    pub is_admin: bool,
}

impl<S> FromRequestParts<S> for CurrentUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        parts
            .extensions
            .get::<CurrentUser>()
            .cloned()
            .ok_or((StatusCode::UNAUTHORIZED, "Unauthorized: Identity missing."))
    }
}

// --- STATE MANAGEMENT ---

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ApiKeyRecord {
    pub name: String,
    pub description: Option<String>,
    pub hash: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UserApiKeys {
    pub email: String,
    pub keys: Vec<ApiKeyRecord>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct AuthStore {
    pub api_keys: HashMap<String, Vec<ApiKeyRecord>>,
    #[serde(skip)]
    pub key_index: HashMap<String, String>,
    #[serde(skip)]
    pub writer_tx: Option<UnboundedSender<UserApiKeys>>,
}

impl AuthStore {
    pub async fn load(db: &Surreal<Any>) -> Result<Self, surrealdb::Error> {
        let mut store = AuthStore::default();

        // Load individual user records
        let mut response = db.query("SELECT * FROM auth_keys").await?;
        let user_keys: Vec<UserApiKeys> = response.take(0)?;
        for uk in user_keys {
            store.api_keys.insert(uk.email, uk.keys);
        }

        // Build the O(1) index on startup
        for (email, records) in &store.api_keys {
            for record in records {
                store.key_index.insert(record.hash.clone(), email.clone());
            }
        }
        Ok(store)
    }

    pub fn generate_key(
        &mut self,
        email: &str,
        name: String,
        description: Option<String>,
    ) -> String {
        let mut key_bytes = [0u8; 32];
        thread_rng().fill_bytes(&mut key_bytes);

        let plaintext_key = format!("sk-{}", STANDARD.encode(key_bytes));
        let hash = hex::encode(Sha256::digest(plaintext_key.as_bytes()));

        self.api_keys
            .entry(email.to_string())
            .or_default()
            .push(ApiKeyRecord {
                name,
                description,
                hash: hash.clone(),
            });

        // Add to O(1) index
        self.key_index.insert(hash, email.to_string());
        self.save_user(email);

        plaintext_key
    }

    pub fn revoke_key(&mut self, email: &str, hash: &str) -> bool {
        if let Some(keys) = self.api_keys.get_mut(email) {
            let initial_len = keys.len();
            keys.retain(|k| k.hash != hash);
            if keys.len() < initial_len {
                self.key_index.remove(hash); // Keep O(1) cache in sync
                self.save_user(email);
                return true;
            }
        }
        false
    }

    pub fn save_user(&self, email: &str) {
        if let Some(keys) = self.api_keys.get(email) {
            if let Some(tx) = &self.writer_tx {
                let payload = UserApiKeys {
                    email: email.to_string(),
                    keys: keys.clone(),
                };
                let _ = tx.send(payload);
            } else {
                error!("[AUTH FAULT] Writer channel missing! Dropping api_keys DB write.");
            }
        }
    }
}

#[derive(Deserialize)]
struct GoogleClientSecret {
    web: GoogleClientSecretWeb,
}

#[derive(Deserialize)]
struct GoogleClientSecretWeb {
    client_id: String,
    client_secret: String,
}

// --- OAUTH2 CLIENT SETUP ---

pub fn build_oauth_client(redirect_uri: &str, secret_path: &str) -> Result<BasicClient, String> {
    // 1. Read the JSON file from the disk
    let file_content = match fs::read_to_string(secret_path) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!(
                "⚠️ CRITICAL: Could not find {} in the manager directory! Error: {}",
                secret_path, e
            );
            error!("{}", msg);
            return Err(msg);
        }
    };

    // 2. Parse the JSON to extract the ID and Secret
    let secret_data: GoogleClientSecret = match serde_json::from_str(&file_content) {
        Ok(d) => d,
        Err(e) => {
            let msg = format!(
                "⚠️ CRITICAL: Failed to parse Google client secret JSON. Make sure it is the 'Web application' format. Error: {}",
                e
            );
            error!("{}", msg);
            return Err(msg);
        }
    };

    let client_id = secret_data.web.client_id;
    let client_secret = secret_data.web.client_secret;

    Ok(BasicClient::new(
        ClientId::new(client_id),
        Some(ClientSecret::new(client_secret)),
        AuthUrl::new("https://accounts.google.com/o/oauth2/v2/auth".to_string()).unwrap(),
        Some(TokenUrl::new("https://oauth2.googleapis.com/token".to_string()).unwrap()),
    )
    // Make sure this matches your Nginx setup exactly! (e.g., https://ai.lan/auth/google/callback)
    .set_redirect_uri(RedirectUrl::new(redirect_uri.to_string()).unwrap()))
}

// --- LOGIN ROUTES ---

pub async fn login_handler(
    session: Session,
    State(state): State<Arc<AppState>>,
) -> Result<Redirect, StatusCode> {
    let (auth_url, csrf_token) = state
        .oauth_client
        .authorize_url(CsrfToken::new_random)
        .add_scope(Scope::new("email".to_string()))
        .url();

    session
        .insert("oauth_csrf_state", csrf_token.secret().clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Redirect::to(auth_url.as_str()))
}

#[derive(Deserialize)]
pub struct AuthRequest {
    pub code: String,
    pub state: String,
}

#[derive(Deserialize)]
pub struct GoogleUser {
    email: String,
}

pub async fn callback_handler(
    session: Session,
    State(state): State<Arc<AppState>>,
    Query(query): Query<AuthRequest>,
) -> Result<Response, StatusCode> {
    let saved_state: Option<String> = session
        .get("oauth_csrf_state")
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if saved_state.is_none() || saved_state.unwrap() != query.state {
        return Err(StatusCode::BAD_REQUEST); // CSRF Attack Detected!
    }

    // map network errors to HTTP status codes
    let token = state
        .oauth_client
        .exchange_code(AuthorizationCode::new(query.code))
        .request_async(async_http_client)
        .await
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    // Use the pooled Reqwest client from AppState
    let user_data = state
        .reqwest_client
        .get("https://www.googleapis.com/oauth2/v2/userinfo")
        .bearer_auth(token.access_token().secret())
        .send()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?
        .json::<GoogleUser>()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let is_admin = state.config.admin_emails.contains(&user_data.email);
    let is_user = state.config.user_emails.contains(&user_data.email);

    if !is_admin && !is_user {
        return Ok((StatusCode::FORBIDDEN, "Email not registered in config.json").into_response());
    }

    session
        .insert("user_email", user_data.email)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Redirect::to("/").into_response())
}

pub async fn logout_handler(session: Session) -> Result<Redirect, StatusCode> {
    session
        .delete()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Redirect::to("/auth/login"))
}

// --- SETTINGS PAGE APIs (Requires Session) ---

#[derive(Deserialize)]
pub struct CreateKeyRequest {
    pub name: String,
    pub description: Option<String>,
}

pub async fn require_session(session: Session) -> Result<String, StatusCode> {
    match session.get::<String>("user_email").await {
        Ok(Some(email)) => Ok(email),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

pub async fn list_keys_handler(
    session: Session,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ApiKeyRecord>>, StatusCode> {
    let email = require_session(session).await?;
    let store = state
        .auth_store
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let keys = store.api_keys.get(&email).cloned().unwrap_or_default();
    Ok(Json(keys))
}

pub async fn create_key_handler(
    session: Session,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateKeyRequest>,
) -> Result<Json<String>, StatusCode> {
    let email = require_session(session).await?;
    let mut store = state
        .auth_store
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let new_key = store.generate_key(&email, payload.name, payload.description);
    Ok(Json(new_key))
}

pub async fn delete_key_handler(
    session: Session,
    Path(hash): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, StatusCode> {
    let email = require_session(session).await?;
    let mut store = state
        .auth_store
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if store.revoke_key(&email, &hash) {
        Ok(StatusCode::OK)
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

// --- DUAL-AUTH MIDDLEWARE (Session OR API Key) ---

pub async fn dual_auth_middleware(
    session: Session,
    State(state): State<Arc<AppState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let mut current_user: Option<CurrentUser> = None;

    if let Ok(Some(email)) = session.get::<String>("user_email").await {
        let is_admin = state.config.admin_emails.contains(&email);
        current_user = Some(CurrentUser { email, is_admin });
    } else if let Some(auth_header) = request
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        && auth_header.starts_with("Bearer ")
    {
        let token = auth_header.trim_start_matches("Bearer ").trim();
        if !token.is_empty() {
            let hash = hex::encode(Sha256::digest(token.as_bytes()));

            let store = state
                .auth_store
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(email) = store.key_index.get(&hash) {
                let is_admin = state.config.admin_emails.contains(email);
                current_user = Some(CurrentUser {
                    email: email.clone(),
                    is_admin,
                });
            }
        }
    }

    // Inject the identity or fail!
    if let Some(user) = current_user {
        request.extensions_mut().insert(user);
        next.run(request).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_store_generate_and_index() {
        let mut store = AuthStore::default();
        let key = store.generate_key("admin@local", "Test Key".into(), None);

        assert!(key.starts_with("sk-"));
        assert_eq!(store.api_keys["admin@local"].len(), 1);
        let hash = &store.api_keys["admin@local"][0].hash;
        assert_eq!(store.key_index.get(hash), Some(&"admin@local".to_string()));
    }

    #[test]
    fn test_auth_store_key_deletion_sync() {
        let mut store = AuthStore::default();
        store.generate_key("test@local", "Key1".into(), None);

        let key2_hash = {
            store.generate_key("test@local", "Key2".into(), None);
            store.api_keys["test@local"].last().unwrap().hash.clone()
        };

        // Simulate the manual deletion route
        let success = store.revoke_key("test@local", &key2_hash);
        assert!(success, "Revoking an existing key should return true");

        assert!(!store.key_index.contains_key(&key2_hash));

        // Deleting a non-existent or already deleted key returns false (which the router maps to 404 NOT FOUND)
        assert!(!store.revoke_key("test@local", &key2_hash));
        assert!(!store.revoke_key("fakeuser@local", "fake_hash"));
    }

    #[test]
    fn test_auth_store_multiple_keys_per_user() {
        let mut store = AuthStore::default();
        let key1 = store.generate_key("user@local", "Key1".into(), None);
        let key2 = store.generate_key("user@local", "Key2".into(), None);

        assert_eq!(store.api_keys["user@local"].len(), 2);
        assert_ne!(key1, key2);

        // Both distinct hashes should reliably resolve to the same user email
        let hash1 = &store.api_keys["user@local"][0].hash;
        let hash2 = &store.api_keys["user@local"][1].hash;
        assert_eq!(store.key_index.get(hash1), Some(&"user@local".to_string()));
        assert_eq!(store.key_index.get(hash2), Some(&"user@local".to_string()));
    }

    #[tokio::test]
    async fn test_current_user_extractor() {
        use axum::extract::FromRequestParts;
        use axum::http::Request;

        // 1. Test missing user extension (Should yield Unauthorized Rejection)
        let req = Request::builder().body(()).unwrap();
        let mut parts = req.into_parts().0;
        let result = CurrentUser::from_request_parts(&mut parts, &()).await;
        assert_eq!(result.unwrap_err().0, StatusCode::UNAUTHORIZED);

        // 2. Test successful extraction
        let mut req = Request::builder().body(()).unwrap();
        req.extensions_mut().insert(CurrentUser {
            email: "admin@local".to_string(),
            is_admin: true,
        });
        let mut parts = req.into_parts().0;
        let result = CurrentUser::from_request_parts(&mut parts, &())
            .await
            .unwrap();
        assert_eq!(result.email, "admin@local");
        assert!(result.is_admin);
    }

    #[tokio::test]
    async fn test_auth_store_surrealdb_flow() {
        // Spin up an isolated, in-memory SurrealDB instance for testing
        let db = surrealdb::engine::any::connect("mem://").await.unwrap();
        db.use_ns("test").use_db("test").await.unwrap();

        let mut store = AuthStore::default();
        store.generate_key("db_test@local", "DB Key".into(), None);

        // Simulate the background task writing the state to the DB
        let user_keys = UserApiKeys {
            email: "db_test@local".to_string(),
            keys: store.api_keys["db_test@local"].clone(),
        };

        let _ = db
            .upsert::<Option<UserApiKeys>>(("auth_keys", "db_test@local"))
            .content(user_keys)
            .await;

        let loaded_store = AuthStore::load(&db).await.unwrap();
        assert_eq!(loaded_store.api_keys.len(), 1);
        assert_eq!(loaded_store.api_keys["db_test@local"][0].name, "DB Key");

        // Verify the O(1) index was correctly rebuilt on load
        let hash = &loaded_store.api_keys["db_test@local"][0].hash;
        assert_eq!(
            loaded_store.key_index.get(hash),
            Some(&"db_test@local".to_string())
        );
    }
}
