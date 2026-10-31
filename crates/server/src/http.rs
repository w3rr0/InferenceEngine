use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, Response, StatusCode};
use serde::Deserialize;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_max_tokens() -> usize {
    512
}

#[derive(Debug)]
pub struct EngineRequest {
    pub prompt: String,
    pub max_tokens: usize,
}

pub type RequestQueue = Arc<Mutex<VecDeque<EngineRequest>>>;

pub async fn handle_request(
    req: Request<hyper::body::Incoming>,
    request_queue: RequestQueue,
) -> Result<Response<Full<Bytes>>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/ping") => {
            let body_bytes = req
                .into_body()
                .collect()
                .await
                .map(|b| b.to_bytes())
                .unwrap_or_default();

            let msg = String::from_utf8_lossy(&body_bytes);
            tracing::info!(msg = %msg, "HTTP ping");

            Ok(Response::new(Full::new(Bytes::from(format!("pong: {msg}")))))
        }

        (&Method::POST, "/chat/completions") => {
            let body_bytes = match req.into_body().collect().await {
                Ok(body) => body.to_bytes(),

                Err(error) => {
                    tracing::error!(%error, "Could not read /generate request body");

                    return Ok(json_error(
                            StatusCode::BAD_REQUEST,
                            "Could not read request body",
                    ));
                }
            };

            let request: GenerateRequest = match serde_json::from_slice(&body_bytes) {
                Ok(request) => request,

                Err(error) => {
                    tracing::error!(%error, "Invalid /generate JSON");

                    return Ok(json_error(
                        StatusCode::BAD_REQUEST,
                        "Invalid JSON body",
                    ));
                }
            };

            if request.prompt.trim().is_empty() {
                return Ok(json_error(
                    StatusCode::BAD_REQUEST,
                    "Prompt cannot be empty",
                ));
            }

            if request.max_tokens == 0 {
                return Ok(json_error(
                    StatusCode::BAD_REQUEST,
                    "max_tokens must be greater than 0",
                ));
            }

            let engine_request = EngineRequest {
                prompt: request.prompt,
                max_tokens: request.max_tokens,
            };
            
            match request_queue.lock() {
                Ok(mut queue) => {
                    queue.push_back(engine_request);
                }
        
                Err(error) => {
                    tracing::error!("Request queue is poisoned: {}", error);
        
                    return Ok(json_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Request queue is unavailable",
                    ));
                }
            }

            let mut response = Response::new(Full::new(Bytes::from(
                r#"{"status":"accepted"}"#,
            )));

            *response.status_mut() = StatusCode::ACCEPTED;
            response.headers_mut().insert(
                hyper::header::CONTENT_TYPE,
                hyper::header::HeaderValue::from_static("application/json"),
            );

            Ok(response)
        }

        // Unknown path
        _ => {
            let mut not_found = Response::new(Full::new(Bytes::from("Not Found")));
            *not_found.status_mut() = StatusCode::NOT_FOUND;
            Ok(not_found)
        }
    }
}

fn json_error(
    status: StatusCode,
    message: &str,
) -> Response<Full<Bytes>> {
    let body = format!(r#"{{"error":"{}"}}"#, message);

    let mut response = Response::new(Full::new(Bytes::from(body)));

    *response.status_mut() = status;

    response.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        hyper::header::HeaderValue::from_static("application/json"),
    );

    response
}