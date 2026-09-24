use crate::memory::BlockTable;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RequestStatus {
    Pending,
    Active,
    Finished,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub block_table: BlockTable,
    pub status: RequestStatus,
}

impl Request {
    pub fn new(id: u64, prompt: Vec<u32>) -> Self {
        Self {
            id,
            prompt,
            generated_tokens: Vec::new(),
            block_table: BlockTable::new(),
            status: RequestStatus::Pending,
        }
    }

    /// Total sequence length (prompt + generated tokens)
    pub fn seq_len(&self) -> usize {
        self.prompt.len() + self.generated_tokens.len()
    }

    /// Returns tokens that must be processed in the next step.
    /// For Active (decode): last generated token.
    /// For Pending (prefill): whole prompt.
    pub fn get_input_tokens(&self) -> Vec<u32> {
        if let Some(&last_token) = self.generated_tokens.last() {
            vec![last_token]
        } else {
            self.prompt.clone()
        }
    }
}