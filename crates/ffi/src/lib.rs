use pyo3::prelude::*;

#[pymodule(name = "core")]
mod core {
    #[pymodule_export]
    use engine_core::engine::Engine;
}