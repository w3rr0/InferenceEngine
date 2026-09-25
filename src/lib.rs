pub mod memory;
pub mod scheduler;
pub mod engine;

use pyo3::prelude::*;

#[pymodule]
mod engine_core {

    #[pymodule_export]
    use super::engine::Engine;
}

/// A Python module implemented in Rust.
#[pymodule]
mod inference_engine {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
