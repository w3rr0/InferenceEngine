use pyo3::prelude::*;

#[pymodule(name = "_bindings")]
mod bindings {
    use pyo3::prelude::*;

    #[pymodule]
    pub mod core {
        #[pymodule_export]
        use engine_core::engine::Engine;
    }
    
    #[pymodule]
    pub mod server {
        #[pymodule_export]
        use ::server::Options;

        #[pymodule_export]
        use ::server::start;
        
        #[pymodule_export]
        use ::server::poll_request;
    }

}