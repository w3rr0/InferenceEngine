use std::net::SocketAddr;

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