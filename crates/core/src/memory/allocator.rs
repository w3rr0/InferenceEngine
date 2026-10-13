use std::collections::VecDeque;
use super::block::{BlockId, BlockTable};

#[derive(Debug, PartialEq)]
pub enum AllocatorError {
    OutOfMemory,
}

pub struct BlockAllocator {
    free_blocks: VecDeque<BlockId>,
    pub block_size: usize,
    pub total_blocks: usize,
}

impl BlockAllocator {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        let free_blocks = (0..total_blocks as BlockId).collect();
        Self {
            free_blocks,
            block_size,
            total_blocks,
        }
    }

    /// Allocates the required number of blocks at once - prefill phase
    pub fn allocate_multiple(&mut self, num_blocks: usize) -> Result<BlockTable, AllocatorError> {
        if self.free_blocks.len() < num_blocks {
            return Err(AllocatorError::OutOfMemory);
        }

        let mut table = BlockTable::new();
        for _ in 0..num_blocks {
            // Bezpieczny unwrap, bo sprawdziliśmy długość wyżej
            table.append(self.free_blocks.pop_front().unwrap());
        }

        Ok(table)
    }

    /// Allocates single block
    pub fn allocate_block(&mut self) -> Result<BlockId, AllocatorError> {
        self.free_blocks.pop_front().ok_or(AllocatorError::OutOfMemory)
    }

    /// Frees the entire array
    pub fn free_table(&mut self, table: &BlockTable) {
        for &block_id in &table.blocks {
            self.free_blocks.push_back(block_id);
        }
    }

    pub fn available_blocks(&self) -> usize {
        self.free_blocks.len()
    }
}