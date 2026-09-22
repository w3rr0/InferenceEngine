use std::collections::HashMap;

/// Memory management errors
#[derive(Debug, PartialEq, Eq)]
pub enum BlockManagerError {
    OutOfMemory,
    SequenceNotFound,
    SequenceAlreadyExists,
}

pub type SequenceId = u64;
pub type PhysicalBlockId = usize;

/// Single sequence memory state
#[derive(Debug, Clone)]
pub struct SequenceState {
    pub block_table: Vec<PhysicalBlockId>,
    pub num_tokens: usize,
}

pub struct BlockManager {
    free_pool: Vec<PhysicalBlockId>,
    sequences: HashMap<SequenceId, SequenceState>,
    block_size: usize,
}

impl BlockManager {
    /// Memory management initialization
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        let free_pool: Vec<PhysicalBlockId> = (0..total_blocks).collect();

        Self {
            free_pool,
            sequences: HashMap::new(),
            block_size,
        }
    }

    /// Allocating starting memory for new prompt in prefill faze
    pub fn allocate(&mut self, seq_id: SequenceId, num_tokens: usize) -> Result<(), BlockManagerError> {
        if self.sequences.contains_key(&seq_id) {
            return Err(BlockManagerError::SequenceAlreadyExists);
        }

        let required_blocks = (num_tokens + self.block_size - 1) / self.block_size;

        if self.free_pool.len() < required_blocks {
            return Err(BlockManagerError::OutOfMemory);
        }

        let mut block_table = Vec::with_capacity(required_blocks);
        for _ in 0..required_blocks {
            block_table.push(self.free_pool.pop().unwrap());
        }

        self.sequences.insert(seq_id, SequenceState {
            block_table,
            num_tokens,
        });

        Ok(())
    }

    /// Register new token generated from model - decode phase
    pub fn append_token(&mut self, seq_id: SequenceId) -> Result<bool, BlockManagerError> {
        let state = self.sequences.get_mut(&seq_id).ok_or(BlockManagerError::SequenceNotFound)?;

        state.num_tokens += 1;

        let capacity = state.block_table.len() * self.block_size;

        if state.num_tokens > capacity {
            if let Some(new_block) = self.free_pool.pop() {
                state.block_table.push(new_block);
                Ok(true)
            } else {
                // If there are no memory left, roll back token counter
                state.num_tokens -= 1;
                Err(BlockManagerError::OutOfMemory)
            }
        } else {
            Ok(false)
        }
    }

    /// Free sequence memory after generating EOS token or terminated connection
    pub fn free_sequence(&mut self, seq_id: SequenceId) -> Result<(), BlockManagerError> {
        if let Some(state) = self.sequences.remove(&seq_id) {
            // Return all physical blocks to the pool
            for block_id in state.block_table {
                self.free_pool.push(block_id);
            }
            Ok(())
        } else {
            Err(BlockManagerError::SequenceNotFound)
        }
    }

    /// Get `block_table` for sequence (used by Scheduler for building batch)
    pub fn get_block_table(&self, seq_id: SequenceId) -> Option<&Vec<PhysicalBlockId>> {
        self.sequences.get(&seq_id).map(|s| &s.block_table)
    }

    /// Return available physical VRAM block count
    pub fn available_blocks(&self) -> usize {
        self.free_pool.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let manager = BlockManager::new(100, 16);
        assert_eq!(manager.available_blocks(), 100);
    }

    #[test]
    fn test_prefill_allocation() {
        let mut manager = BlockManager::new(100, 16);

        manager.allocate(1, 35).unwrap();

        assert_eq!(manager.available_blocks(), 97);
        let table = manager.get_block_table(1).unwrap();
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn test_decode_append_token() {
        let mut manager = BlockManager::new(100, 16);

        manager.allocate(1, 15).unwrap();
        assert_eq!(manager.get_block_table(1).unwrap().len(), 1);

        let newly_allocated = manager.append_token(1).unwrap();
        assert!(!newly_allocated);
        assert_eq!(manager.get_block_table(1).unwrap().len(), 1);

        let newly_allocated = manager.append_token(1).unwrap();
        assert!(newly_allocated);
        assert_eq!(manager.get_block_table(1).unwrap().len(), 2);
        assert_eq!(manager.available_blocks(), 98);
    }

    #[test]
    fn test_free_sequence() {
        let mut manager = BlockManager::new(100, 16);

        manager.allocate(1, 40).unwrap();
        assert_eq!(manager.available_blocks(), 97);

        manager.free_sequence(1).unwrap();
        assert_eq!(manager.available_blocks(), 100);
        assert_eq!(manager.get_block_table(1), None);
    }

    #[test]
    fn test_out_of_memory() {
        let mut manager = BlockManager::new(2, 16);

        let result = manager.allocate(1, 35);
        assert_eq!(result, Err(BlockManagerError::OutOfMemory));

        assert_eq!(manager.available_blocks(), 2);
    }
}
