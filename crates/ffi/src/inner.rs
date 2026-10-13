use pyo3::prelude::*;
use engine_core::engine::Engine as CoreEngine;

#[pyclass(name = "Engine", unsendable)]
pub struct PyEngine {
    inner: CoreEngine,
}

#[pymethods]
impl PyEngine {
    #[new]
    pub fn new(total_blocks: usize, block_size: usize, max_batch_size: usize) -> Self {
        Self {
            inner: CoreEngine::new(total_blocks, block_size, max_batch_size),
        }
    }

    pub fn add_request(&mut self, prompt: Vec<u32>, max_tokens: usize, eos_token_id: u32) {
        self.inner.add_request(prompt, max_tokens, eos_token_id);
    }

    pub fn has_unfinished_requests(&self) -> bool {
        self.inner.has_unfinished_requests()
    }

    pub fn get_next(&mut self) -> Option<(Vec<u64>, Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<usize>, bool)> {
        self.inner.get_next()
    }

    pub fn step(&mut self, request_ids: Vec<u64>, new_tokens: Vec<u32>) {
        self.inner.step(request_ids, new_tokens);
    }
}