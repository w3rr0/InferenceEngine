pub type BlockId = u32;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockTable {
    pub blocks: Vec<BlockId>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    pub fn append(&mut self, block_id: BlockId) {
        self.blocks.push(block_id);
    }
}