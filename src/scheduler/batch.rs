#[derive(Debug)]
pub struct Batch {
    pub request_ids: Vec<u64>,
    pub input_tokens: Vec<Vec<u32>>,
    pub block_tables: Vec<Vec<u32>>,
    pub seq_lens: Vec<usize>,
    pub is_prefill: bool,
}
