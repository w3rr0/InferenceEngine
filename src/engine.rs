use pyo3::prelude::*;
use crate::memory::BlockAllocator;
use crate::scheduler::{Scheduler, Request};

#[pyclass(unsendable)]
pub struct Engine {
    allocator: BlockAllocator,
    scheduler: Scheduler,
    request_counter: u64,
}

#[pymethods]
impl Engine {
    #[new]
    pub fn new(total_blocks: usize, block_size: usize, max_batch_size: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(total_blocks, block_size),
            scheduler: Scheduler::new(max_batch_size),
            request_counter: 0,
        }
    }

    /// Temporary method simulating the arrival of a new request
    pub fn add_dummy_request(&mut self, prompt: Vec<u32>) {
        self.request_counter += 1;
        let req = Request::new(self.request_counter, prompt);
        self.scheduler.add_request(req);
    }

    /// Returns a batch from the scheduler: (request_ids, input_tokens, block_tables, seq_lens, is_prefill)
    pub fn get_next(&mut self) -> Option<(Vec<u64>, Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<usize>, bool)> {
        if let Some(batch) = self.scheduler.next_batch(&mut self.allocator) {
            Some((
                batch.request_ids,
                batch.input_tokens,
                batch.block_tables,
                batch.seq_lens,
                batch.is_prefill,
            ))
        } else {
            None
        }
    }

    /// Receives generated tokens from PyTorch model
    pub fn step(&mut self, request_ids: Vec<u64>, new_tokens: Vec<u32>) {
        for (id, token) in request_ids.into_iter().zip(new_tokens.into_iter()) {
            if let Some(req) = self.scheduler.active.iter_mut().find(|r| r.id == id) {
                req.generated_tokens.push(token);

                // TODO: For now logic for adding blocks (`allocator.allocate_block()`) and preemption during OOM in `decode` is skipped
            }
        }
    }
}