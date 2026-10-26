mod config;
mod error;
mod http;
mod grpc;
mod server;

pub use config::Options;
pub use error::{Error, Result};
pub use server::start;


pub mod pb {
    tonic::include_proto!("inference");
}

#[cfg(feature = "python")]
pub use server::poll_request;

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .try_init();
}