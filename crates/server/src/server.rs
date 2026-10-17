use crate::grpc::GenerationImpl;
use crate::http::handle_request;
use crate::pb::generation_server::GenerationServer;
use crate::Result;

use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use std::net::SocketAddr;
use tokio::net::TcpListener;

pub async fn start_grpc(addr: SocketAddr) -> Result<()> {
    tracing::info!("Starting GRPC server on {}", addr);
    let service = GenerationImpl::default();

    tonic::transport::Server::builder()
        .add_service(GenerationServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

pub async fn start_http(addr: SocketAddr) -> Result<()> {
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