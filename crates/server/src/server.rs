use crate::grpc::GenerationImpl;
use crate::http::handle_request;
use crate::pb::generation_server::GenerationServer;
use crate::{Options, Result};

use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use std::net::SocketAddr;
use tokio::net::TcpListener;

async fn start_grpc(addr: SocketAddr) -> Result<()> {
    tracing::info!("Starting GRPC server on {}", addr);
    let service = GenerationImpl::default();

    tonic::transport::Server::builder()
        .add_service(GenerationServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

async fn start_http(addr: SocketAddr) -> Result<()> {
    tracing::info!("Starting HTTP server on {}", addr);
    let listener = TcpListener::bind(addr).await?;

    loop {
        let (tcp_stream, _) = listener.accept().await?;
        let io = TokioIo::new(tcp_stream);

        tokio::task::spawn(async move {
            let service = service_fn(handle_request);
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

    let grpc_task = tokio::spawn(async move {
        if let Err(e) = start_grpc(grpc_addr).await {
            tracing::error!("gRPC server finished with error: {}", e);
        }
    });

    let http_task = tokio::spawn(async move {
        if let Err(e) = start_http(http_addr).await {
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