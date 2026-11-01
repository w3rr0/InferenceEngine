use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Frame};
use hyper::{Method, Request, Response, StatusCode};
use serde::Deserialize;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use http_body_util::combinators::BoxBody;
use tokio::sync::mpsc;
use tonic::codegen::tokio_stream;
use uuid::Uuid;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

const MEMORY_BUFFER_SIZE: usize = 128;

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
    pub id: Uuid,
    pub prompt: String,
    pub max_tokens: usize,
    pub sender: mpsc::Sender<GenerationChunk>,
}

#[derive(Debug)]
pub struct GenerationChunk {
    pub text: String,
    pub is_finished: bool,
}


pub type RequestQueue = Arc<Mutex<VecDeque<EngineRequest>>>;

pub async fn handle_request(
    req: Request<hyper::body::Incoming>,
    request_queue: RequestQueue,
) -> Result<Response<BoxBody<Bytes, Infallible>>, hyper::Error> {
    let request_id = Uuid::new_v4();

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

            Ok(Response::new(Full::new(Bytes::from(format!("pong: {msg}"))).boxed()))
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

            let (tx, rx) = mpsc::channel::<GenerationChunk>(MEMORY_BUFFER_SIZE);

            let engine_request = EngineRequest {
                id: request_id.clone(),
                prompt: request.prompt,
                max_tokens: request.max_tokens,
                sender: tx,
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


            let stream = ReceiverStream::new(rx).map(|chunk| {
                let mut sse_msg = String::new();
                if !chunk.text.is_empty() {
                    let json = serde_json::json!({"choices": [{"delta": {"content": chunk.text}}]});
                    sse_msg.push_str(&format!("data: {}\n\n", json));
                }
                if chunk.is_finished {
                    sse_msg.push_str("data: [DONE]\n\n");
                }
                Ok::<_, Infallible>(Frame::data(Bytes::from(sse_msg)))
            });

            let body = http_body_util::StreamBody::new(stream).boxed();

            let mut response = Response::new(body);
            *response.status_mut() = StatusCode::OK;
            response.headers_mut().insert(
                hyper::header::CONTENT_TYPE,
                hyper::header::HeaderValue::from_static("text/event-stream"),
            );
            response.headers_mut().insert(
                hyper::header::CACHE_CONTROL,
                hyper::header::HeaderValue::from_static("no-cache"),
            );
            response.headers_mut().insert(
                hyper::header::CONNECTION,
                hyper::header::HeaderValue::from_static("keep-alive"),
            );

            Ok(response)
        }

        // Unknown path
        _ => {
            let body = Full::new(Bytes::from("Not Found")).boxed();
            let mut not_found = Response::new(body);
            *not_found.status_mut() = StatusCode::NOT_FOUND;
            Ok(not_found)
        }
    }
}

fn json_error(
    status: StatusCode,
    message: &str,
) -> Response<BoxBody<Bytes, Infallible>> {
    let json_string = format!(r#"{{"error":"{}"}}"#, message);
    let body = Full::new(Bytes::from(json_string)).boxed();

    let mut response = Response::new(body);
    *response.status_mut() = status;

    response.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        hyper::header::HeaderValue::from_static("application/json"),
    );

    response
}