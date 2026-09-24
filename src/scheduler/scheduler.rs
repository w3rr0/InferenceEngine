use std::collections::VecDeque;
use super::request::{Request, RequestStatus};
use super::batch::Batch;
use crate::memory::BlockAllocator;

pub struct Scheduler {
    pub pending: VecDeque<Request>,
    pub active: Vec<Request>,
    pub max_batch_size: usize,
}

impl Scheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            active: Vec::new(),
            max_batch_size,
        }
    }

    pub fn add_request(&mut self, request: Request) {
        self.pending.push_back(request);
    }

    /// Retrieves the most urgent requests and forms a `Batch` from them
    pub fn next_batch(&mut self, allocator: &mut BlockAllocator) -> Option<Batch> {
        // First, we continue with what is already being generated (Decode)
        if !self.active.is_empty() {
            return Some(self.build_decode_batch(allocator));
        }

        // If there is nothing in active, we check for new requests (Prefill)
        if !self.pending.is_empty() {
            return Some(self.build_prefill_batch(allocator));
        }

        None
    }

    fn build_prefill_batch(&mut self, allocator: &mut BlockAllocator) -> Batch {
        let mut batch_reqs = Vec::new();

        // Pull up to N requests from the pending queue
        while batch_reqs.len() < self.max_batch_size && !self.pending.is_empty() {
            if let Some(mut req) = self.pending.pop_front() {
                // Calculate how many blocks the prompt requires
                let required_blocks = (req.prompt.len() + allocator.block_size - 1) / allocator.block_size;

                // Try to allocate VRAM
                if let Ok(table) = allocator.allocate_multiple(required_blocks) {
                    req.block_table = table;
                    req.status = RequestStatus::Active;
                    batch_reqs.push(req);
                } else {
                    // OOM - Move the query to the front of the queue
                    self.pending.push_front(req);
                    break;
                }
            }
        }

        self.construct_batch(batch_reqs, true)
    }

    fn build_decode_batch(&mut self, _allocator: &mut BlockAllocator) -> Batch {
        // TODO: In the full version, we would check here whether the request exceeded its block's capacity and allocate a new one using `allocator.allocate_block()`.

        // Extract up to max_batch_size from active
        let count = std::cmp::min(self.active.len(), self.max_batch_size);
        let batch_reqs: Vec<Request> = self.active.drain(..count).collect();

        self.construct_batch(batch_reqs, false)
    }

    fn construct_batch(&mut self, reqs: Vec<Request>, is_prefill: bool) -> Batch {
        let mut request_ids = Vec::new();
        let mut input_tokens = Vec::new();
        let mut block_tables = Vec::new();
        let mut seq_lens = Vec::new();

        for req in reqs {
            request_ids.push(req.id);
            input_tokens.push(req.get_input_tokens());
            block_tables.push(req.block_table.blocks.clone());
            seq_lens.push(req.seq_len());

            // Put the request back into the active queue
            self.active.push(req);
        }

        Batch {
            request_ids,
            input_tokens,
            block_tables,
            seq_lens,
            is_prefill,
        }
    }
}