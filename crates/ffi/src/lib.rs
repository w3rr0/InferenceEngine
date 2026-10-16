use pyo3::prelude::*;

#[pymodule(name = "_bindings")]
mod bindings {
    #[pymodule_export]
    use engine_core::engine::Engine;
}