use axum::{
    async_trait,
    extract::{Path, Query, Request, State, FromRequestParts},
    http::{StatusCode, request::Parts},
    middleware::Next,
    response::{IntoResponse, Redirect, Response},
    Json,
};
use oauth2::{
    basic::BasicClient, reqwest::async_http_client, AuthUrl, AuthorizationCode, ClientId,
    ClientSecret, CsrfToken, RedirectUrl, Scope, TokenResponse, TokenUrl,
};
use rand::{thread_rng, RngCore};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, sync::Arc};
use tokio::sync::mpsc::UnboundedSender;
use tower_sessions::Session;
use crate::AppState;

#[derive(Clone, Debug)]
pub struct CurrentUser {
    pub email: String,
    pub is_admin: bool,
}

#[async_trait]
impl<S> FromRequestParts<S> for CurrentUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        parts.extensions.get::<CurrentUser>()
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

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct AuthStore {
    pub api_keys: HashMap<String, Vec<ApiKeyRecord>>, 
    #[serde(skip)] 
    pub key_index: HashMap<String, String>,
    #[serde(skip)]
    pub writer_tx: Option<UnboundedSender<String>>,
}

impl AuthStore {
    pub fn load() -> Self {
        let mut store = if let Ok(data) = fs::read_to_string("api_keys.json") {
            match serde_json::from_str(&data) {
                Ok(s) => s,
                Err(e) => {
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let backup_name = format!("api_keys_{}.json.bak", timestamp);
                    eprintln!("⚠️ CRITICAL: api_keys.json is corrupted! Error: {}", e);
                    eprintln!("⚠️ Backing up corrupted file to {} to prevent data loss.", backup_name);
                    let _ = fs::rename("api_keys.json", backup_name);
                    AuthStore { api_keys: HashMap::new(), key_index: HashMap::new(), writer_tx: None }
                }
            }
        } else {
            AuthStore { api_keys: HashMap::new(), key_index: HashMap::new(), writer_tx: None } 
        };
        
        // Build the O(1) index on startup
        for (email, records) in &store.api_keys {
            for record in records {
                store.key_index.insert(record.hash.clone(), email.clone());
            }
        }
        store
    }

    pub fn generate_key(&mut self, email: &str, name: String, description: Option<String>) -> String {
        let mut key_bytes = [0u8; 32];
        thread_rng().fill_bytes(&mut key_bytes);
        
        let plaintext_key = format!("sk-{}", STANDARD.encode(key_bytes));
        let hash = hex::encode(Sha256::digest(plaintext_key.as_bytes()));
        
        self.api_keys.entry(email.to_string()).or_default().push(ApiKeyRecord {
            name, description, hash: hash.clone(),
        });
        
        // Add to O(1) index
        self.key_index.insert(hash, email.to_string());
        self.save();
        
        plaintext_key
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Some(tx) = &self.writer_tx {
                let _ = tx.send(json);
            } else {
                eprintln!("⚠️ [AUTH FAULT] Writer channel missing! Dropping api_keys.json write.");
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

pub fn build_oauth_client(redirect_uri: &str) -> BasicClient {
    // 1. Read the JSON file from the disk
    let file_content = fs::read_to_string("client_secret.apps.googleusercontent.com.json")
        .expect("⚠️ CRITICAL: Could not find client_secret.apps.googleusercontent.com.json in the manager directory!");
    
    // 2. Parse the JSON to extract the ID and Secret
    let secret_data: GoogleClientSecret = serde_json::from_str(&file_content)
        .expect("⚠️ CRITICAL: Failed to parse Google client secret JSON. Make sure it is the 'Web application' format.");

    let client_id = secret_data.web.client_id;
    let client_secret = secret_data.web.client_secret;

    BasicClient::new(
        ClientId::new(client_id),
        Some(ClientSecret::new(client_secret)),
        AuthUrl::new("https://accounts.google.com/o/oauth2/v2/auth".to_string()).unwrap(),
        Some(TokenUrl::new("https://oauth2.googleapis.com/token".to_string()).unwrap()),
    )
    // Make sure this matches your Nginx setup exactly! (e.g., https://ai.lan/auth/google/callback)
    .set_redirect_uri(RedirectUrl::new(redirect_uri.to_string()).unwrap())
}

// --- LOGIN ROUTES ---

pub async fn login_handler(
    session: Session,
    State(state): State<Arc<AppState>>
) -> Result<Redirect, StatusCode> {
    let (auth_url, csrf_token) = state.oauth_client
        .authorize_url(CsrfToken::new_random)
        .add_scope(Scope::new("email".to_string()))
        .url();

    session.insert("oauth_csrf_state", csrf_token.secret().clone())
        .await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Redirect::to(auth_url.as_str()))
}

#[derive(Deserialize)]
pub struct AuthRequest {
    pub code: String,
    pub state: String,
}

#[derive(Deserialize)]
pub struct GoogleUser { email: String }

pub async fn callback_handler(
    session: Session,
    State(state): State<Arc<AppState>>,
    Query(query): Query<AuthRequest>,
) -> Result<Response, StatusCode> { 
    
    let saved_state: Option<String> = session.get("oauth_csrf_state")
        .await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    if saved_state.is_none() || saved_state.unwrap() != query.state {
        return Err(StatusCode::BAD_REQUEST); // CSRF Attack Detected!
    }

    // map network errors to HTTP status codes
    let token = state.oauth_client.exchange_code(AuthorizationCode::new(query.code))
        .request_async(async_http_client)
        .await
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    // Use the pooled Reqwest client from AppState
    let user_data = state.reqwest_client
        .get("https://www.googleapis.com/oauth2/v2/userinfo")
        .bearer_auth(token.access_token().secret())
        .send().await.map_err(|_| StatusCode::BAD_GATEWAY)?
        .json::<GoogleUser>().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let is_admin = state.config.admin_emails.contains(&user_data.email);
    let is_user = state.config.user_emails.contains(&user_data.email);

    if !is_admin && !is_user {
        return Ok((StatusCode::FORBIDDEN, "Email not registered in config.json").into_response());
    }

    session.insert("user_email", user_data.email).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Redirect::to("/").into_response())
}

pub async fn logout_handler(session: Session) -> Result<Redirect, StatusCode> {
    session.delete().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
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
    let store = state.auth_store.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let keys = store.api_keys.get(&email).cloned().unwrap_or_default();
    Ok(Json(keys))
}

pub async fn create_key_handler(
    session: Session,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateKeyRequest>, 
) -> Result<Json<String>, StatusCode> {
    let email = require_session(session).await?;
    let mut store = state.auth_store.lock().unwrap_or_else(|poisoned| poisoned.into_inner()); 
    let new_key = store.generate_key(&email, payload.name, payload.description);
    Ok(Json(new_key))
}

pub async fn delete_key_handler(
    session: Session,
    Path(hash): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, StatusCode> {
    let email = require_session(session).await?;
    let mut store = state.auth_store.lock().unwrap_or_else(|poisoned| poisoned.into_inner()); 
    if let Some(keys) = store.api_keys.get_mut(&email) {
        keys.retain(|k| k.hash != hash);
        store.key_index.remove(&hash); // Keep O(1) cache in sync
        store.save();
    }
    Ok(StatusCode::OK)
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
    } 
    else if let Some(auth_header) = request.headers().get("Authorization").and_then(|h| h.to_str().ok()) && auth_header.starts_with("Bearer ") {
        let token = auth_header.trim_start_matches("Bearer ").trim();
        if !token.is_empty() {
            let hash = hex::encode(Sha256::digest(token.as_bytes()));

            let store = state.auth_store.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(email) = store.key_index.get(&hash) {
                let is_admin = state.config.admin_emails.contains(email);
                current_user = Some(CurrentUser { email: email.clone(), is_admin });
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