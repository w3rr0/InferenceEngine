use crate::pb;
use tonic::{Request, Response, Status};

#[derive(Default)]
pub(crate) struct GenerationImpl;

#[tonic::async_trait]
impl pb::generation_server::Generation for GenerationImpl {
    async fn ping(
        &self,
        req: Request<pb::PingRequest>,
    ) -> Result<Response<pb::PingResponse>, Status> {
        let msg = req.into_inner().msg;
        tracing::info!(msg = %msg, "gRPC ping");
        Ok(Response::new(pb::PingResponse {
            msg: format!("pong: {msg}"),
        }))
    }
}