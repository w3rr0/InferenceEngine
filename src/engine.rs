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
    pub fn add_request(&mut self, prompt: Vec<u32>, max_tokens: usize, eos_token_id: u32) {
        self.request_counter += 1;
        let req = Request::new(self.request_counter, prompt, max_tokens, eos_token_id);
        self.scheduler.add_request(req);
    }

    /// Returns `true` if there are unfinished tasks
    pub fn has_unfinished_requests(&self) -> bool {
        !self.scheduler.active.is_empty() || !self.scheduler.pending.is_empty()
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
        // Save generated token
        for (id, token) in request_ids.into_iter().zip(new_tokens.into_iter()) {
            if let Some(req) = self.scheduler.active.iter_mut().find(|r| r.id == id) {
                req.generated_tokens.push(token);
            }
        }

        // Manage memory and life cycle
        let mut i = 0;
        while i < self.scheduler.active.len() {
            // If: generation finished
            if self.scheduler.active[i].is_finished() {
                let mut req = self.scheduler.active.remove(i);
                req.status = crate::scheduler::RequestStatus::Finished;

                self.allocator.free_table(&req.block_table);
                continue;
            }

            // If: Check if new block is required for new token
            let seq_len = self.scheduler.active[i].seq_len();
            let block_size = self.allocator.block_size;

            if seq_len > self.scheduler.active[i].block_table.blocks.len() * block_size {
                if let Ok(new_block) = self.allocator.allocate_block() {
                    self.scheduler.active[i].block_table.append(new_block);
                    i += 1;
                } else {
                    // If: No VRAM
                    let last_idx = self.scheduler.active.len() - 1;
                    let mut preempted = self.scheduler.active.remove(last_idx);

                    // Return free blocks
                    self.allocator.free_table(&preempted.block_table);
                    preempted.block_table.blocks.clear();

                    preempted.generated_tokens.clear();
                    preempted.status = crate::scheduler::RequestStatus::Pending;
                    self.scheduler.pending.insert(0, preempted);
                }
            } else {
                i += 1;
            }
        }
    }
}