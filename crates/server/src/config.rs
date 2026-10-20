use std::net::SocketAddr;

#[cfg(feature = "python")]
use pyo3::{
    exceptions::PyValueError,
    prelude::*
};

#[cfg_attr(feature = "python", pyclass(module = "inference_engine.server", from_py_object))]
#[derive(Debug, Clone)]
pub struct Options {
    pub http_addr: SocketAddr,
    pub grpc_addr: SocketAddr,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            http_addr: "0.0.0.0:8080".parse().unwrap(),
            grpc_addr: "0.0.0.0:50051".parse().unwrap(),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Options {
    #[new]
    #[pyo3(signature = (
    http_addr = "0.0.0.0:8080",
    grpc_addr = "0.0.0.0:50051",
    ))]
    pub fn _new(http_addr: &str, grpc_addr: &str) -> PyResult<Self> {
        Ok(Self {
            http_addr: http_addr
                .parse()
                .map_err(|err| PyValueError::new_err(format!("Invalid HTTP address: {err}")))?,
            grpc_addr: grpc_addr
                .parse()
                .map_err(|err| PyValueError::new_err(format!("Invalid gRPC address: {err}")))?,
        })
    }
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn http_addr(mut self, addr: impl Into<SocketAddr>) -> Self {
        self.http_addr = addr.into();
        self
    }

    pub fn grpc_addr(mut self, addr: impl Into<SocketAddr>) -> Self {
        self.grpc_addr = addr.into();
        self
    }
}