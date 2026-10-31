use crate::grpc::GenerationImpl;
use crate::http::{handle_request, EngineRequest, RequestQueue};
use crate::pb::generation_server::GenerationServer;
use crate::{Options, Result};

use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::net::TcpListener;
use uuid::Uuid;

static REQUEST_QUEUE: OnceLock<RequestQueue> = OnceLock::new();

async fn start_grpc(addr: SocketAddr) -> Result<()> {
    tracing::info!("Starting GRPC server on {}", addr);
    let service = GenerationImpl::default();

    tonic::transport::Server::builder()
        .add_service(GenerationServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

async fn start_http(
    addr: SocketAddr,
    request_queue: RequestQueue,
) -> Result<()> {
    tracing::info!("Starting HTTP server on {}", addr);
    let listener = TcpListener::bind(addr).await?;

    loop {
        let (tcp_stream, _) = listener.accept().await?;
        let io = TokioIo::new(tcp_stream);

        let request_queue = request_queue.clone();

        tokio::task::spawn(async move {
            let service = service_fn(move |request| {
                handle_request(request, request_queue.clone())
            });
            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service)
                .await {
                tracing::error!("HTTP server error: {}", err);
            }
        });
    }
}

async fn run(options: Options) -> Result<()> {
    let http_addr = options.http_addr;
    let grpc_addr = options.grpc_addr;

    let request_queue: RequestQueue =
        Arc::new(Mutex::new(VecDeque::new()));

    let _ = REQUEST_QUEUE.set(request_queue.clone());

    let http_queue = request_queue.clone();

    let grpc_task = tokio::spawn(async move {
        if let Err(e) = start_grpc(grpc_addr).await {
            tracing::error!("gRPC server finished with error: {}", e);
        }
    });

    let http_task = tokio::spawn(async move {
        if let Err(e) = start_http(http_addr, http_queue).await {
            tracing::error!("HTTP server finished with error: {}", e);
        }
    });

    tokio::try_join!(grpc_task, http_task)?;

    Ok(())
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
#[pyo3(signature = (options = None))]
pub fn start(options: Option<Options>) -> pyo3::PyResult<()> {
    let options = options.unwrap_or_default();

    std::thread::Builder::new()
        .name("inference-server".to_owned())
        .spawn(move || {
            crate::init_tracing();

            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(err) => {
                    tracing::error!("Could not create Tokio runtime: {}", err);
                    return;
                }
            };

            if let Err(err) = runtime.block_on(run(options)) {
                tracing::error!("Inference server stopped with error: {}", err);
            }
        })
        .map(|_| ())
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
}

#[cfg(not(feature = "python"))]
pub use run as start;

#[cfg(feature = "python")]
#[pyo3::pyfunction]
pub fn poll_request() -> Option<(Uuid, String, usize)> {
    let queue = REQUEST_QUEUE.get()?;

    let mut queue = queue.lock().ok()?;
    let request = queue.pop_front()?;

    Some((request.id, request.prompt, request.max_tokens))
}