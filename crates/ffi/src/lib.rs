pub mod inner;

use pyo3::prelude::*;

#[pymodule]
mod engine_core {
    #[pymodule_export]
    use super::inner::PyEngine as Engine;
}