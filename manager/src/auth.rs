use axum::{
    extract::{Path, Query, Request, State},
    http::StatusCode,
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
use tower_sessions::Session;
use crate::AppState;

// --- STATE MANAGEMENT ---

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ApiKeyRecord {
    pub name: String,
    pub description: Option<String>,
    pub hash: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct AuthStore {
    pub allowed_emails: Vec<String>,
    pub api_keys: HashMap<String, Vec<ApiKeyRecord>>, 
}

impl AuthStore {
    pub fn load() -> Self {
        let allowed = fs::read_to_string("allowed_emails.txt")
            .unwrap_or_default()
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut store = if let Ok(data) = fs::read_to_string("api_keys.json") {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            AuthStore { allowed_emails: Vec::new(), api_keys: HashMap::new() }
        };
        store.allowed_emails = allowed;
        store
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.api_keys) {
            let _ = fs::write("api_keys.json", json);
        }
    }

    pub fn generate_key(&mut self, email: &str, name: String, description: Option<String>) -> String {
        let mut key_bytes = [0u8; 32];
        thread_rng().fill_bytes(&mut key_bytes); // <-- Fixed random generator
        
        let plaintext_key = format!("sk-{}", STANDARD.encode(key_bytes)); // <-- Fixed base64
        let hash = hex::encode(Sha256::digest(plaintext_key.as_bytes()));
        
        self.api_keys.entry(email.to_string()).or_default().push(ApiKeyRecord {
            name,
            description,
            hash,
        });
        self.save();
        
        plaintext_key
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

pub fn build_oauth_client() -> BasicClient {
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
    .set_redirect_uri(RedirectUrl::new("http://localhost:3000/auth/google/callback".to_string()).unwrap())
}

// --- LOGIN ROUTES ---

pub async fn login_handler() -> impl IntoResponse {
    let client = build_oauth_client();
    let (auth_url, _csrf_token) = client
        .authorize_url(CsrfToken::new_random)
        .add_scope(Scope::new("email".to_string()))
        .url();
    Redirect::to(auth_url.as_str())
}

#[derive(Deserialize)]
pub struct AuthRequest { code: String }

#[derive(Deserialize)]
pub struct GoogleUser { email: String }

pub async fn callback_handler(
    session: Session,
    State(state): State<Arc<AppState>>,
    Query(query): Query<AuthRequest>,
) -> impl IntoResponse {
    let client = build_oauth_client();
    let token = match client.exchange_code(AuthorizationCode::new(query.code)).request_async(async_http_client).await {
        Ok(t) => t,
        Err(_) => return (StatusCode::UNAUTHORIZED, "Failed to exchange token").into_response(),
    };

    let req_client = reqwest::Client::new();
    let user_data = req_client
        .get("https://www.googleapis.com/oauth2/v2/userinfo")
        .bearer_auth(token.access_token().secret())
        .send().await.unwrap()
        .json::<GoogleUser>().await.unwrap();

    // Scope the lock so it drops BEFORE the `.await` point below
    let is_allowed = {
        let store = state.auth_store.lock().unwrap();
        store.allowed_emails.contains(&user_data.email)
    };

    if !is_allowed {
        return (StatusCode::FORBIDDEN, "Email not on allowed list.").into_response();
    }

    session.insert("user_email", user_data.email).await.unwrap();
    Redirect::to("/").into_response()
}

pub async fn logout_handler(session: Session) -> impl IntoResponse {
    session.delete().await.unwrap();
    Redirect::to("/auth/login")
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
    let store = state.auth_store.lock().unwrap();
    let keys = store.api_keys.get(&email).cloned().unwrap_or_default();
    Ok(Json(keys))
}

pub async fn create_key_handler(
    session: Session,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateKeyRequest>, 
) -> Result<Json<String>, StatusCode> {
    let email = require_session(session).await?;
    let mut store = state.auth_store.lock().unwrap(); 
    let new_key = store.generate_key(&email, payload.name, payload.description);
    Ok(Json(new_key))
}

pub async fn delete_key_handler(
    session: Session,
    Path(hash): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, StatusCode> {
    let email = require_session(session).await?;
    let mut store = state.auth_store.lock().unwrap();
    if let Some(keys) = store.api_keys.get_mut(&email) {
        keys.retain(|k| k.hash != hash); // <-- Check the struct's hash field
        store.save();
    }
    Ok(StatusCode::OK)
}

// --- DUAL-AUTH MIDDLEWARE (Session OR API Key) ---

pub async fn dual_auth_middleware(
    session: Session,
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    // Try internal UI authentication first (Session Cookie)
    if session.get::<String>("user_email").await.unwrap_or(None).is_some() {
        return next.run(request).await;
    }

    let mut is_authorized = false;

    // Fallback to external API authentication (Bearer Token)
    if let Some(auth_header) = request.headers().get("Authorization").and_then(|h| h.to_str().ok()) {
        if auth_header.starts_with("Bearer ") {
            let token = auth_header.trim_start_matches("Bearer ").trim();
            // Hash the incoming token to compare against our secure storage
            let hash = hex::encode(Sha256::digest(token.as_bytes()));

            {
                let store = state.auth_store.lock().unwrap();
                for keys in store.api_keys.values() {
                    if keys.iter().any(|k| k.hash == hash) {
                        is_authorized = true;
                        break;
                    }
                }
            } // Lock is safely dropped here
        }
    }
    
    if is_authorized {
        next.run(request).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}