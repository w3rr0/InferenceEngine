#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("grpc transport: {0}")]
    GrpcTransport(#[from] tonic::transport::Error),

    #[error("task join: {0}")]
    Join(#[from] tokio::task::JoinError),
}

pub type Result<T> = std::result::Result<T, Error>;