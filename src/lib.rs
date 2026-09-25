pub mod memory;
pub mod scheduler;
pub mod engine;

use pyo3::prelude::*;

#[pymodule]
mod engine_core {
    #[pymodule_export]
    use super::engine::Engine;
}