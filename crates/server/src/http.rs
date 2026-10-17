use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, Response, StatusCode};
use std::convert::Infallible;

pub async fn handle_request(
    req: Request<hyper::body::Incoming>,
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

        // Unknown path
        _ => {
            let mut not_found = Response::new(Full::new(Bytes::from("Not Found")));
            *not_found.status_mut() = StatusCode::NOT_FOUND;
            Ok(not_found)
        }
    }
}