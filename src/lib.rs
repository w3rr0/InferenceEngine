pub mod memory;
pub mod scheduler;
pub mod engine;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "engine_core")]
mod inference_engine {

    #[pymodule_export]
    use super::engine::Engine;
}